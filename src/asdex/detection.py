"""Jacobian and Hessian sparsity detection via jaxpr graph analysis."""

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from asdex._interpret import prop_jaxpr
from asdex._interpret._commons import empty_index_sets, offset_identity_index_sets
from asdex.pattern import SparsityPattern

def _normalize_args(arg_shapes, argnums):
    """Normalize input shapes and argnums into standardized tuples."""
    argnums_tup = (argnums,) if isinstance(argnums, int) else tuple(argnums)

    # Infer if arg_shapes represents a single argument or multiple
    # shapes = x -> ((x))
    if isinstance(arg_shapes, int):
        shapes = ((arg_shapes,),)
    # shapes = (x, y, z) -> ((x, y, z))
    elif isinstance(arg_shapes, tuple) and all(isinstance(x, int) for x in arg_shapes):
        shapes = (arg_shapes,)
    # shapes = (x, y, (a, b, c)) -> ((x), (y), (a, b, c))
    else:
        shapes = tuple((s,) if isinstance(s, int) else tuple(s) for s in arg_shapes)

    return shapes, argnums_tup

def jacobian_sparsity(
    f: Callable,
    arg_shapes: int | tuple[int, ...] | tuple[tuple[int, ...], ...] | list,
    *,
    argnums: int | tuple[int, ...] = 0,
) -> SparsityPattern:
    """Detect global Jacobian sparsity pattern.

    Analyzes the computation graph structure directly, without evaluating
    any derivatives. The result is valid for all inputs.

    Args:
        f: Function taking an array and returning an array.
        arg_shapes: Shape(s) of the input array(s).
        argnums: Integer or tuple of integers indicating which arguments
            to track dependencies for.

    Returns:
        SparsityPattern containing the fused horizontal block matrix
        representing all tracked arguments.
    """

    shapes, argnums_tup = _normalize_args(arg_shapes, argnums)

    dummy_inputs = [jnp.zeros(shape) for shape in shapes]
    closed_jaxpr, out_struct = jax.make_jaxpr(f, return_shape=True)(*dummy_inputs)
    jaxpr = closed_jaxpr.jaxpr

    # flattened function output length essentially
    m = math.prod(out_struct.shape) if out_struct.shape else 1

    # dummy_input = jnp.zeros(input_shape)
    # closed_jaxpr = jax.make_jaxpr(f)(dummy_input)
    # jaxpr = closed_jaxpr.jaxpr
    # m = int(jax.eval_shape(f, dummy_input).size)
    # n = input_shape if isinstance(input_shape, int) else math.prod(input_shape)

    input_indices = []
    current_offset = 0

    # Seed the tracked arguments with offset identities
    # Builds a list of lists of sets containing the indexes into each arg in argnums for output tracking
    for i, shape in enumerate(shapes):
        n = shape if isinstance (shape, int) else math.prod(shape) # flattened input shape of each input
        if i in argnums_tup:
            input_indices.append(offset_identity_index_sets(n, current_offset))
            current_offset += n
        else:
            input_indices.append(empty_index_sets(n))
    total_tracked_size = current_offset # length of flattened inputs included in argnums

    # Initialize: input element i depends on input index i
    # input_indices = [identity_index_sets(n)]

    # Build state_consts from closed jaxpr consts for static index tracking
    state_consts = {
        var: np.asarray(val)
        for var, val in zip(jaxpr.constvars, closed_jaxpr.consts, strict=False)
    }

    # Propagate through the jaxpr
    output_indices_list = prop_jaxpr(jaxpr, input_indices, state_consts)

    # Flatten all outputs to build a single composite sparse matrix
    out_indices = []
    for out_deps in output_indices_list:
        out_indices.extend(out_deps)

    # Extract output dependencies (first output variable)
    # out_indices = output_indices_list[0] if output_indices_list else []

    # Build sparsity pattern
    rows = []
    cols = []
    for i, deps in enumerate(out_indices):
        for j in deps:
            rows.append(i)
            cols.append(j)

    # shape_tuple = (input_shape,) if isinstance(input_shape, int) else tuple(input_shape)
    # return SparsityPattern.from_coo(rows, cols, (m, n), input_shape=shape_tuple)

    return SparsityPattern.from_coo(
        rows,
        cols,
        (m, total_tracked_size),
        input_shape=(total_tracked_size,),
    )

def hessian_sparsity(
    f: Callable, input_shape: int | tuple[int, ...]
) -> SparsityPattern:
    """Detect global Hessian sparsity pattern for f: R^n -> R.

    Analyzes the Jacobian sparsity of the gradient function,
    without evaluating any derivatives.
    The result is valid for all inputs.

    If ``f`` returns a squeezable shape like ``(1,)`` or ``(1, 1)``,
    it is automatically squeezed to scalar.

    Args:
        f: Scalar-valued function taking an array.
        input_shape: Shape of the input array.
            An integer is treated as a 1D length.

    Returns:
        SparsityPattern of shape ``(n, n)``
            where ``n = prod(input_shape)``.
            Entry ``(i, j)`` is present if ``H[i, j]`` may be nonzero.
    """
    f = _ensure_scalar(f, input_shape)
    return jacobian_sparsity(jax.grad(f), input_shape)


def _ensure_scalar(f: Callable, input_shape: int | tuple[int, ...]) -> Callable:
    """Ensure ``f`` returns a scalar, auto-squeezing if possible.

    If ``f`` already returns shape ``()``, it is returned unchanged.
    If squeezing the output yields a scalar (e.g. shape ``(1,)``),
    a wrapped version is returned.
    Otherwise, raises ``ValueError``.
    """
    out = jax.eval_shape(f, jnp.zeros(input_shape))
    if out.shape == ():
        return f
    squeezed = jax.eval_shape(lambda x: jnp.squeeze(f(x)), jnp.zeros(input_shape))
    if squeezed.shape != ():
        raise ValueError(
            f"Expected scalar-valued function, but f has output shape {out.shape}."
        )
    return lambda x: jnp.squeeze(f(x))
