"""Shared utilities for performance testing and benchmarking."""

import time
import tracemalloc
import jax
import jax.numpy as jnp
import sympy as sp

def tridiagonal_target(x, y):
    """
    Highly sparse, banded Jacobian. 
    Output i depends on x[i-1], x[i], x[i+1] and y[i].
    """
    x_right = jnp.roll(x, 1)
    x_left = jnp.roll(x, -1)
    return x_left * x + x_right * jnp.sin(y)

def sympy_setup_and_lambdify(N):
    """
    Replicates the target function using SymPy symbolics, computes the
    symbolic Jacobian, and compiles it down to a fast NumPy C-extension.
    """
    x_syms = sp.symbols(f'x0:{N}')
    y_syms = sp.symbols(f'y0:{N}')
    
    exprs = []
    for i in range(N):
        # Modulo arithmetic replicates the exact behavior of jnp.roll
        left = x_syms[(i - 1) % N]
        right = x_syms[(i + 1) % N]
        expr = left * x_syms[i] + right * sp.sin(y_syms[i])
        exprs.append(expr)
        
    F = sp.Matrix(exprs)
    X = sp.Matrix(x_syms)
    
    # Compute symbolic Jacobian
    J = F.jacobian(X)
    
    # Lambdify converts the SymPy matrix of formulas into a fast NumPy function
    return sp.lambdify((x_syms, y_syms), J, modules='numpy')

def get_xla_execution_memory(jitted_fn, *args):
    """Queries the XLA compiler for the exact device memory footprint."""
    lowered = jitted_fn.lower(*args)
    compiled = lowered.compile()
    stats = compiled.memory_analysis()
    
    if stats is not None:
        total_bytes = (stats.temp_size_in_bytes + 
                       stats.argument_size_in_bytes + 
                       stats.output_size_in_bytes - 
                       stats.alias_size_in_bytes)
        return total_bytes / (1024 * 1024)
    return 0.0

def track_execution_time(func, *args, runs=10):
    """Helper to track pure execution time."""
    t0 = time.perf_counter()
    for _ in range(runs):
        res = func(*args)
        if hasattr(res, 'block_until_ready'):
            res.block_until_ready()
    return (time.perf_counter() - t0) / runs

def track_execution(func, *args, runs=10):
    """Helper to track execution time and peak Python memory (via tracemalloc)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    
    for _ in range(runs):
        res = func(*args)
        # Force JAX to finish async execution before stopping the clock
        if hasattr(res, 'block_until_ready'):
            res.block_until_ready()
            
    total_time = time.perf_counter() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return total_time / runs, peak_mem / (1024 * 1024) # Return seconds and MB