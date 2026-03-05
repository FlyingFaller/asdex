"""Tests for multi-argument Jacobian functionality using Sympy for verification."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp

import asdex


def eval_sympy_jacobian(func, argnums, *inputs):
    """
    Evaluates the exact dense Jacobian of a function using Sympy.
    The function `func` must be written using standard python operators
    (+, -, *, **, @) so it can accept both JAX arrays and Numpy object 
    arrays containing Sympy symbols.
    """
    argnums_tup = (argnums,) if isinstance(argnums, int) else tuple(argnums)
    
    # Create symbolic inputs
    sym_args = []
    flat_syms_list = []
    for i, arr in enumerate(inputs):
        # Create flat symbols x_i_0, x_i_1, ... and reshape to match input
        flat_syms = sp.symbols(f'x_{i}_0:{arr.size}')
        sym_args.append(np.array(flat_syms, dtype=object).reshape(arr.shape))
        flat_syms_list.append(flat_syms)
        
    # Evaluate function symbolically
    sym_out = func(*sym_args)
    sym_out_flat = np.array(sym_out).flatten()
    
    # Map symbols to their numeric values
    subs_dict = {}
    for i, arr in enumerate(inputs):
        for sym, val in zip(flat_syms_list[i], arr.flatten()):
            subs_dict[sym] = float(val)
            
    results = []
    for argnum in argnums_tup:
        # Compute exact symbolic jacobian
        jac = sp.Matrix(sym_out_flat).jacobian(sp.Matrix(flat_syms_list[argnum]))
        # Substitute numeric values
        jac_num = np.array(jac.evalf(subs=subs_dict)).astype(float)
        results.append(jac_num)
        
    return results[0] if isinstance(argnums, int) else tuple(results)


def test_original_single_argument():
    """Verify original asdex behavior is completely unaffected."""
    x = jnp.array([1.0, 2.0, 3.0])
    
    def f(x):
        return x**2 + 2*x
        
    jac_fn = asdex.jacobian(f, x.shape, argnums=0)
    asdex_jac = jac_fn(x).todense()
    
    sympy_jac = eval_sympy_jacobian(f, 0, x)
    np.testing.assert_allclose(asdex_jac, sympy_jac, atol=1e-5)


def test_multiple_args_single_argnum():
    """Test differentiating with respect to a single argument out of many."""
    x = jnp.array([1.0, 2.0])
    y = jnp.array([3.0, 4.0, 5.0])
    
    # f(x, y) outputs a vector of size 3
    def f(x, y):
        return x[0] * y + x[1]**2
        
    # Track only 'y' (argnums=1)
    jac_fn = asdex.jacobian(f, [x.shape, y.shape], argnums=1)
    asdex_jac_y = jac_fn(x, y).todense()
    
    sympy_jac_y = eval_sympy_jacobian(f, 1, x, y)
    
    assert asdex_jac_y.shape == (3, 3)
    np.testing.assert_allclose(asdex_jac_y, sympy_jac_y, atol=1e-5)


def test_multiple_args_tuple_argnums():
    """Test differentiating with respect to multiple arguments simultaneously."""
    x = jnp.array([1.0, 2.0])
    y = jnp.array([3.0, 4.0])
    z = jnp.array([5.0])
    
    def f(x, y, z):
        return x**2 + 3*x*y + y**3 + z
        
    jac_fn = asdex.jacobian(f, [x.shape, y.shape, z.shape], argnums=(0, 1))
    asdex_jac_x, asdex_jac_y = jac_fn(x, y, z)
    
    sympy_jac_x, sympy_jac_y = eval_sympy_jacobian(f, (0, 1), x, y, z)
    
    np.testing.assert_allclose(asdex_jac_x.todense(), sympy_jac_x, atol=1e-5)
    np.testing.assert_allclose(asdex_jac_y.todense(), sympy_jac_y, atol=1e-5)


def test_multidimensional_args():
    """Test multi-dimensional arrays with tuple argnums and matrix mult."""
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]]) # 2x2
    y = jnp.array([[2.0, 0.0], [0.0, 2.0]]) # 2x2
    z = jnp.array([1.0, 2.0])               # 2,
    
    def f(x, y, z):
        # x @ y is matrix multiplication. Output is flattened.
        out = x @ y
        return out.flatten() + z.sum()
        
    # Track x (0) and z (2)
    jac_fn = asdex.jacobian(f, [x.shape, y.shape, z.shape], argnums=(0, 2))
    asdex_jac_x, asdex_jac_z = jac_fn(x, y, z)
    
    sympy_jac_x, sympy_jac_z = eval_sympy_jacobian(f, (0, 2), x, y, z)
    
    np.testing.assert_allclose(asdex_jac_x.todense(), sympy_jac_x, atol=1e-5)
    np.testing.assert_allclose(asdex_jac_z.todense(), sympy_jac_z, atol=1e-5)


def test_zero_dependence_arg():
    """Test when an argument is tracked but doesn't affect the output."""
    x = jnp.array([1.0, 2.0])
    y = jnp.array([3.0, 4.0])
    
    def f(x, y):
        # y is entirely ignored
        return x**2
        
    jac_fn = asdex.jacobian(f, [x.shape, y.shape], argnums=(0, 1))
    asdex_jac_x, asdex_jac_y = jac_fn(x, y)
    
    sympy_jac_x, sympy_jac_y = eval_sympy_jacobian(f, (0, 1), x, y)
    
    np.testing.assert_allclose(asdex_jac_x.todense(), sympy_jac_x, atol=1e-5)
    # The jacobian wrt y should be a matrix of pure zeros
    np.testing.assert_allclose(asdex_jac_y.todense(), sympy_jac_y, atol=1e-5)
    assert asdex_jac_y.nse == 0