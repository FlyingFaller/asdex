"""Simple runner for multi-argument sparse Jacobian detection and decompression."""

import jax
import jax.numpy as jnp
import numpy as np

# Adjust these imports if your folder structure is different
from asdex.decompression import jacobian
from asdex.detection import jacobian_sparsity

def sample_target_fn(x, y, z):
    """
    A non-linear function with 3 inputs and multiple cross-dependencies.
    x shape: (2,) | y shape: (3,) | z shape: (2,)
    Output shape: (4,)
    """
    return jnp.stack([
        x[0] * y[0],              # Depends on x[0], y[0]
        x[1] * jnp.sin(z[0]),     # Depends on x[1], z[0]
        y[1] * z[1] + x[0],       # Depends on x[0], y[1], z[1]
        z[1] ** 2                 # Depends on z[1]
    ])

def run_multi_arg_tests():
    print("--- Starting Multi-Argument Sparse Jacobian Tests ---\n")
    
    # 1. Setup
    shapes = ((2,), (3,), (2,))
    argnums = (0, 2) # We want the Jacobian with respect to x and z
    
    x = jnp.array([1.5, -0.5])
    y = jnp.array([2.0, 1.0, -1.0])
    z = jnp.array([0.5, 3.0])

    # 2. Test Sparsity Detection
    print("Testing Phase 1: Sparsity Detection...")
    pattern = jacobian_sparsity(sample_target_fn, shapes, argnums=argnums)
    
    if pattern.shape == (4, 4) and pattern.input_shape == (4,):
        print("  [PASS] Fused pattern shape is correct: (4, 4)")
    else:
        print(f"  [FAIL] Expected pattern shape (4, 4), got {pattern.shape}")
        return

    # 3. Test Dense Baseline
    print("\nTesting Phase 2: Computing Dense JAX Baseline...")
    dense_jac_x = jax.jacobian(sample_target_fn, argnums=0)(x, y, z)
    dense_jac_z = jax.jacobian(sample_target_fn, argnums=2)(x, y, z)
    print("  [PASS] Dense baseline computed.")

    # 4. Test Sparse Compilation & JIT
    print("\nTesting Phase 3: Sparse Jacobian JIT Compilation...")
    sparse_jac_fn = jacobian(sample_target_fn, shapes, argnums=argnums)
    
    try:
        jitted_sparse_jac_fn = jax.jit(sparse_jac_fn)
        # Run it once to trigger compilation
        sparse_jac_x, sparse_jac_z = jitted_sparse_jac_fn(x, y, z)
        print("  [PASS] JIT compilation successful (No dynamic shape errors!).")
    except Exception as e:
        print(f"  [FAIL] JIT compilation crashed:\n{e}")
        return

    # 5. Test Mathematical Correctness
    print("\nTesting Phase 4: Mathematical Correctness...")
    
    x_match = np.allclose(sparse_jac_x.todense(), dense_jac_x, atol=1e-5)
    z_match = np.allclose(sparse_jac_z.todense(), dense_jac_z, atol=1e-5)
    
    if x_match and z_match:
        print("  [PASS] Sparse BCOO matrices match dense JAX outputs perfectly.")
        print("\n--- All Tests Passed! Engine is ready. ---")
    else:
        print("  [FAIL] Math mismatch detected between sparse and dense computation.")
        if not x_match:
            print("         Arg 0 (x) failed.")
        if not z_match:
            print("         Arg 2 (z) failed.")

if __name__ == "__main__":
    # Ensure JAX runs on CPU for standard testing to avoid GPU initialization delays
    jax.config.update("jax_platform_name", "cpu")
    run_multi_arg_tests()