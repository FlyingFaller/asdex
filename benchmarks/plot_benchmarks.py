"""Unified benchmark for Dense JAX, Sparse Asdex, and Symbolic SymPy."""

import time
import tracemalloc
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# Adjusted import for being in the asdex/benchmark/ folder
# import sys
# from pathlib import Path
# sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from asdex.decompression import jacobian

# Import shared utilities
from shared_utils import (
    tridiagonal_target,
    sympy_setup_and_lambdify,
    get_xla_execution_memory,
    track_execution_time
)

def run_benchmarks(Ns, runs=10):
    metrics = {
        'N': Ns,
        'dense_sc_time': [], 'dense_sc_mem': [],
        'dense_ex_time': [], 'dense_ex_mem': [],
        'sparse_sc_time': [], 'sparse_sc_mem': [],
        'sparse_ex_time': [], 'sparse_ex_mem': [],
        'sympy_sc_time': [], 'sympy_sc_mem': [],
        'sympy_ex_time': [],
    }
    
    for N in Ns:
        print(f"\nBenchmarking N={N}...")
        shapes = ((N,), (N,))
        argnums = 0
        
        x = jnp.linspace(0.1, 1.0, N)
        y = jnp.linspace(-1.0, 1.0, N)
        
        x_np = np.linspace(0.1, 1.0, N)
        y_np = np.linspace(-1.0, 1.0, N)
        
        # ==========================================
        # 1. DENSE JAX BENCHMARK
        # ==========================================
        tracemalloc.start()
        t0 = time.perf_counter()
        
        dense_jac_fn = jax.jacfwd(tridiagonal_target, argnums=argnums)
        jitted_dense = jax.jit(dense_jac_fn)
        _ = jitted_dense(x, y).block_until_ready()
        
        sc_time = time.perf_counter() - t0
        _, py_sc_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        metrics['dense_sc_time'].append(sc_time)
        metrics['dense_sc_mem'].append(py_sc_mem / (1024 * 1024))
        
        ex_time = track_execution_time(jitted_dense, x, y, runs=runs)
        ex_mem = get_xla_execution_memory(jitted_dense, x, y)
        metrics['dense_ex_time'].append(ex_time)
        metrics['dense_ex_mem'].append(ex_mem)
        
        # ==========================================
        # 2. SPARSE ASDEX BENCHMARK
        # ==========================================
        tracemalloc.start()
        t0 = time.perf_counter()
        
        sparse_jac_fn = jacobian(tridiagonal_target, shapes, argnums=argnums)
        jitted_sparse = jax.jit(sparse_jac_fn)
        _ = jitted_sparse(x, y).block_until_ready()
        
        sc_time = time.perf_counter() - t0
        _, py_sc_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        metrics['sparse_sc_time'].append(sc_time)
        metrics['sparse_sc_mem'].append(py_sc_mem / (1024 * 1024))
        
        ex_time = track_execution_time(jitted_sparse, x, y, runs=runs)
        ex_mem = get_xla_execution_memory(jitted_sparse, x, y)
        metrics['sparse_ex_time'].append(ex_time)
        metrics['sparse_ex_mem'].append(ex_mem)
        
        # ==========================================
        # 3. SYMPY BENCHMARK
        # ==========================================
        if N <= 250:
            tracemalloc.start()
            t0 = time.perf_counter()
            sympy_fn = sympy_setup_and_lambdify(N)
            sc_time = time.perf_counter() - t0
            _, py_sc_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            ex_time = track_execution_time(sympy_fn, x_np, y_np, runs=runs)
            metrics['sympy_sc_time'].append(sc_time)
            metrics['sympy_sc_mem'].append(py_sc_mem / (1024 * 1024))
            metrics['sympy_ex_time'].append(ex_time)
            print(f"  SymPy  -> Exec: {ex_time:.5f}s")
        else:
            # Mask out values for Matplotlib so the line stops drawing
            metrics['sympy_sc_time'].append(np.nan)
            metrics['sympy_sc_mem'].append(np.nan)
            metrics['sympy_ex_time'].append(np.nan)

        print(f"  Dense  -> Exec: {metrics['dense_ex_time'][-1]:.5f}s | XLA Mem: {metrics['dense_ex_mem'][-1]:.2f} MB")
        print(f"  Sparse -> Exec: {metrics['sparse_ex_time'][-1]:.5f}s | XLA Mem: {metrics['sparse_ex_mem'][-1]:.2f} MB")
        
    return metrics

def plot_metrics(metrics):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Performance Benchmark: Dense JAX vs. Sparse Asdex vs. SymPy', fontsize=16, fontweight='bold')
    Ns = metrics['N']
    
    dense_style, dense_color = 'o-', '#d62728'  # Red
    sparse_style, sparse_color = 's-', '#1f77b4' # Blue
    sympy_style, sympy_color = '^--', '#2ca02c'  # Green

    # Subplot 1: Setup Time
    ax = axs[0, 0]
    ax.plot(Ns, metrics['dense_sc_time'], dense_style, color=dense_color, label='Dense JAX (Trace+JIT)')
    ax.plot(Ns, metrics['sparse_sc_time'], sparse_style, color=sparse_color, label='Sparse Asdex (Detect+Color+JIT)')
    ax.plot(Ns, metrics['sympy_sc_time'], sympy_style, color=sympy_color, label='SymPy (Derive+Lambdify)')
    ax.set_title('Setup + Compile Time', fontsize=14)
    ax.set_ylabel('Time (Seconds)')
    ax.set_xlabel('Input Shape')
    
    # Cap y-axis based on JAX/Asdex maximums to avoid SymPy distortion
    max_sc_time = max(max(metrics['dense_sc_time']), max(metrics['sparse_sc_time']), min(metrics['sympy_sc_time']))
    ax.set_ylim(bottom=0, top=max_sc_time * 1.1)
    
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    # Subplot 2: Setup Python Memory
    ax = axs[0, 1]
    ax.plot(Ns, metrics['dense_sc_mem'], dense_style, color=dense_color, label='Dense JAX')
    ax.plot(Ns, metrics['sparse_sc_mem'], sparse_style, color=sparse_color, label='Sparse Asdex')
    ax.plot(Ns, metrics['sympy_sc_mem'], sympy_style, color=sympy_color, label='SymPy')
    ax.set_title('Setup + Compile Memory Peak (Python Allocation)', fontsize=14)
    ax.set_ylabel('Memory (MB)')
    ax.set_xlabel('Input Shape')
    
    # Cap y-axis based on JAX/Asdex maximums
    max_sc_mem = max(max(metrics['dense_sc_mem']), max(metrics['sparse_sc_mem']), min(metrics['sympy_sc_mem']))
    ax.set_ylim(bottom=0, top=max_sc_mem * 1.1)
    
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    # Subplot 3: Execution Time (Log Scale)
    ax = axs[1, 0]
    ax.plot(Ns, metrics['dense_ex_time'], dense_style, color=dense_color, label='Dense JAX')
    ax.plot(Ns, metrics['sparse_ex_time'], sparse_style, color=sparse_color, label='Sparse Asdex')
    ax.plot(Ns, metrics['sympy_ex_time'], sympy_style, color=sympy_color, label='SymPy Lambdify (NumPy)')
    ax.set_title('Execution Time per Call (Log Scale)', fontsize=14)
    ax.set_ylabel('Time (Seconds)')
    ax.set_xlabel('Input Shape')
    ax.set_yscale('log')
    
    # Log scale uses multiplicative padding instead of additive
    max_ex_time = max(max(metrics['dense_ex_time']), max(metrics['sparse_ex_time']), min(metrics['sympy_ex_time']))
    ax.set_ylim(top=max_ex_time * 2.0)
    
    ax.grid(True, which="both", linestyle='--', alpha=0.6)
    ax.legend()

    # Subplot 4: Execution Device Memory (Log Scale)
    ax = axs[1, 1]
    ax.plot(Ns, metrics['dense_ex_mem'], dense_style, color=dense_color, label='Dense JAX')
    ax.plot(Ns, metrics['sparse_ex_mem'], sparse_style, color=sparse_color, label='Sparse Asdex')
    # SymPy executes in standard NumPy on CPU so it doesn't use XLA device memory, skipped here.
    ax.set_title('Execution XLA Device Memory (Log Scale)', fontsize=14)
    ax.set_ylabel('Memory (MB)')
    ax.set_xlabel('Input Shape')
    ax.set_yscale('log')
    
    max_ex_mem = max(max(metrics['dense_ex_mem']), max(metrics['sparse_ex_mem']))
    if max_ex_mem > 0:
        ax.set_ylim(top=max_ex_mem * 2.0)
        
    ax.grid(True, which="both", linestyle='--', alpha=0.6)
    ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plt.show()

if __name__ == "__main__":
    jax.config.update("jax_platform_name", "cpu")
    # Using a slightly denser array of Ns at the start to capture the SymPy drop-off accurately
    test_Ns = [50, 100, 150, 200, 250, 500, 1000, 2500, 4000, 5000, 10000]
    results = run_benchmarks(test_Ns, runs=10)
    plot_metrics(results)