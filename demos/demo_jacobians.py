"""Visual demonstration of multi-argument sparse Jacobians vs Sympy exact Jacobians."""

import time
import math
import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp
import asdex


def build_sympy_jacobian_callable(func, argnums, input_shapes):
    """
    Evaluates the exact dense Jacobian symbolically and compiles it 
    into a fast numerical callable using SymPy's lambdify.
    """
    argnums_tup = (argnums,) if isinstance(argnums, int) else tuple(argnums)
    
    sym_args = []
    flat_syms_list = []
    all_flat_syms = []
    
    # Create symbolic inputs based strictly on shapes
    for i, shape in enumerate(input_shapes):
        size = math.prod(shape) if shape else 1
        flat_syms = sp.symbols(f'x_{i}_0:{size}')
        sym_args.append(np.array(flat_syms, dtype=object).reshape(shape))
        flat_syms_list.append(flat_syms)
        all_flat_syms.extend(flat_syms)
        
    # Evaluate function symbolically
    sym_out = func(*sym_args)
    sym_out_flat = np.array(sym_out).flatten()
    
    # Formulate symbolic jacobian matrices
    jac_matrices = []
    for argnum in argnums_tup:
        jac = sp.Matrix(sym_out_flat).jacobian(sp.Matrix(flat_syms_list[argnum]))
        jac_matrices.append(jac)
        
    # Compile into a fast numpy-backed C-level evaluation function
    lambdified_func = sp.lambdify(all_flat_syms, jac_matrices, modules='numpy')
    
    # Wrap it to handle incoming unflattened arrays naturally
    def callable_wrapper(*inputs):
        flat_inputs = []
        for arr in inputs:
            flat_inputs.extend(np.array(arr).flatten())
            
        evaled = lambdified_func(*flat_inputs)
        results = [np.array(res).astype(float) for res in evaled]
        return results[0] if isinstance(argnums, int) else tuple(results)
        
    return callable_wrapper


def main():
    # 1. Define the math function
    def f(x, y, z):
        return [
            x[0]**2 * y[1],
            y[0]**3 + z[0] / 2.0,
            x[1] * z[1] - x[0]
        ]
        
    def f_jax(x, y, z):
        return jnp.array(f(x, y, z))

    # We only know the shapes upfront
    shapes = [(2,), (2,), (2,)]
    argnums = (0, 1, 2)
    
    print("--- 1. Formulation & Compilation Phase ---")
    
    # Formulate ASDEX Callable
    t0 = time.perf_counter()
    jac_fn_asdex_raw = asdex.jacobian(f_jax, shapes, argnums=argnums)
    
    # @jax.jit <---- Does not work... 
    def jac_fn_asdex(*args):
        return jac_fn_asdex_raw(*args)
    t_asdex_formulate = time.perf_counter() - t0
    print(f"ASDEX formulation time: {t_asdex_formulate:.4f} seconds")

    # Formulate Sympy Callable
    t0 = time.perf_counter()
    jac_fn_sympy = build_sympy_jacobian_callable(f, argnums, shapes)
    t_sympy_formulate = time.perf_counter() - t0
    print(f"SymPy formulation time: {t_sympy_formulate:.4f} seconds\n")


    print("--- 2. Execution Phase ---")
    N_RUNS = 5
    asdex_times = []
    sympy_times = []
    
    # Generate some initial dummy data strictly for JAX compilation warm-up
    x_w, y_w, z_w = np.random.randn(2), np.random.randn(2), np.random.randn(2)
    warmup_res = jac_fn_asdex(x_w, y_w, z_w)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), warmup_res)

    last_asdex_res = None
    last_sympy_res = None
    last_inputs = None

    for i in range(N_RUNS):
        # New data arrives
        x_val = np.random.randn(2)
        y_val = np.random.randn(2)
        z_val = np.random.randn(2)
        last_inputs = (x_val, y_val, z_val)
        
        # Time ASDEX (must block to ensure async JAX GPU/CPU ops are finished)
        t0 = time.perf_counter()
        sparse_jacs = jac_fn_asdex(x_val, y_val, z_val)
        jax.tree_util.tree_map(lambda arr: arr.block_until_ready(), sparse_jacs)
        asdex_times.append(time.perf_counter() - t0)
        last_asdex_res = sparse_jacs
        
        # Time SymPy lambdified execution
        t0 = time.perf_counter()
        dense_jacs = jac_fn_sympy(x_val, y_val, z_val)
        sympy_times.append(time.perf_counter() - t0)
        last_sympy_res = dense_jacs

    # Print execution timing
    print(f"Executed {N_RUNS} randomized runs.")
    print(f"ASDEX avg execution time: {np.mean(asdex_times)*1000:.4f} ms")
    print(f"SymPy avg execution time: {np.mean(sympy_times)*1000:.4f} ms\n")


    print("--- 3. Results Verification (Last Run) ---")
    print(f"Inputs:\nx: {last_inputs[0]}\ny: {last_inputs[1]}\nz: {last_inputs[2]}\n")
    
    asdex_dense_x = last_asdex_res[0].todense()
    asdex_dense_y = last_asdex_res[1].todense()
    asdex_dense_z = last_asdex_res[2].todense()

    sym_dense_x, sym_dense_y, sym_dense_z = last_sympy_res

    def print_side_by_side(title, mat_asdex, mat_sym):
        print(f"=== {title} ===")
        print(f"{'ASDEX (Dense)':<25} | {'Sympy (Exact)':<25}")
        print("-" * 53)
        for row_a, row_s in zip(mat_asdex, mat_sym):
            str_a = np.array2string(row_a, precision=2, floatmode='fixed', suppress_small=True)
            str_s = np.array2string(row_s, precision=2, floatmode='fixed', suppress_small=True)
            print(f"{str_a:<25} | {str_s:<25}")
        print("\n")

    print_side_by_side("Jacobian wrt x", asdex_dense_x, sym_dense_x)
    print_side_by_side("Jacobian wrt y", asdex_dense_y, sym_dense_y)
    print_side_by_side("Jacobian wrt z", asdex_dense_z, sym_dense_z)


if __name__ == "__main__":
    main()