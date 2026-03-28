"""Console-based benchmark comparing Dense JAX, Sparse Asdex, and SymPy."""

import time
import tracemalloc
import jax
import jax.numpy as jnp
import numpy as np

# import sys
# from pathlib import Path
# Adjust path to import asdex properly from the benchmark folder
# sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from asdex.decompression import jacobian

from shared_utils import (
    tridiagonal_target, 
    sympy_setup_and_lambdify, 
    track_execution,
    track_execution_time,
    get_xla_execution_memory
)

def benchmark(N=100, runs=10):
    print(f"\n" + "="*58)
    print(f"  BENCHMARKING N={N} (Matrix Size: {N}x{N}) OVER {runs} RUNS")
    print("="*58)
    
    shapes = ((N,), (N,))
    argnums = 0
    
    # JAX inputs
    x_jax = jnp.linspace(0.1, 1.0, N)
    y_jax = jnp.linspace(-1.0, 1.0, N)
    
    # NumPy inputs (for SymPy lambdify)
    x_np = np.linspace(0.1, 1.0, N)
    y_np = np.linspace(-1.0, 1.0, N)

    # ==========================================
    # 1. DENSE JAX BENCHMARK
    # ==========================================
    print("\n--- 1. Dense JAX ---")
    
    dense_jac_fn = jax.jacfwd(tridiagonal_target, argnums=argnums)
    jitted_dense = jax.jit(dense_jac_fn)
    
    # Compile
    tracemalloc.start()
    t0 = time.perf_counter()
    _ = jitted_dense(x_jax, y_jax).block_until_ready()
    dense_compile_time = time.perf_counter() - t0
    _, dense_compile_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"  Compile Time:   {dense_compile_time:.4f} sec  | Peak Mem: {dense_compile_mem/(1024*1024):.2f} MB")
    
    # Execute (Using XLA Memory)
    dense_run_time = track_execution_time(jitted_dense, x_jax, y_jax, runs=runs)
    dense_run_mem = get_xla_execution_memory(jitted_dense, x_jax, y_jax)
    print(f"  Execution Time: {dense_run_time:.6f} sec  | XLA Mem: {dense_run_mem:.2f} MB")

    # ==========================================
    # 2. SPARSE ASDEX BENCHMARK
    # ==========================================
    print("\n--- 2. Sparse Asdex ---")
    
    tracemalloc.start()
    t0 = time.perf_counter()
    sparse_jac_fn = jacobian(tridiagonal_target, shapes, argnums=argnums)
    setup_time = time.perf_counter() - t0
    _, setup_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  Setup Time:     {setup_time:.4f} sec  | Peak Mem: {setup_mem/(1024*1024):.2f} MB")
    
    jitted_sparse = jax.jit(sparse_jac_fn)
    
    # Compile
    tracemalloc.start()
    t0 = time.perf_counter()
    _ = jitted_sparse(x_jax, y_jax).block_until_ready()
    sparse_compile_time = time.perf_counter() - t0
    _, sparse_compile_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"  Compile Time:   {sparse_compile_time:.4f} sec  | Peak Mem: {sparse_compile_mem/(1024*1024):.2f} MB")
    
    # Execute (Using XLA Memory)
    sparse_run_time = track_execution_time(jitted_sparse, x_jax, y_jax, runs=runs)
    sparse_run_mem = get_xla_execution_memory(jitted_sparse, x_jax, y_jax)
    print(f"  Execution Time: {sparse_run_time:.6f} sec  | XLA Mem: {sparse_run_mem:.2f} MB")

    # ==========================================
    # 3. SYMPY BENCHMARK
    # ==========================================
    print("\n--- 3. SymPy (Symbolic) ---")
    
    if N > 250:
        print(f"  [SKIPPED] N={N} is too large for SymPy.")
        sympy_run_time = float('inf')
    else:
        # Setup (Symbolic derivation)
        tracemalloc.start()
        t0 = time.perf_counter()
        sympy_fn = sympy_setup_and_lambdify(N)
        sympy_setup_time = time.perf_counter() - t0
        _, sympy_setup_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"  Derive Time:    {sympy_setup_time:.4f} sec  | Peak Mem: {sympy_setup_mem/(1024*1024):.2f} MB")
        
        # Execute (Using tracemalloc since SymPy runs in standard NumPy)
        sympy_run_time, sympy_run_mem = track_execution(sympy_fn, x_np, y_np, runs=runs)
        print(f"  Execution Time: {sympy_run_time:.6f} sec  | Peak Mem: {sympy_run_mem:.2f} MB")


if __name__ == "__main__":
    # Force CPU for consistent benchmarking 
    jax.config.update("jax_platform_name", "cpu")
    
    # 1. The small benchmark where SymPy can compete
    benchmark(N=50, runs=100)
    
    # 2. The medium benchmark where Sparse AD starts to pull ahead
    benchmark(N=200, runs=50)

    # 3. The large benchmark to demonstrate scaling (SymPy disabled automatically)
    benchmark(N=2500, runs=10)