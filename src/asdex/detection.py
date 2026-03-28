"""Jacobian and Hessian sparsity detection via jaxpr graph analysis."""

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from asdex._interpret import prop_jaxpr
from asdex._interpret._commons import empty_index_sets, offset_identity_index_sets
from asdex.pattern import SparsityPattern


def jacobian_sparsity(
    f: Callable, 
    input_shape: tuple[int, ...] | tuple[tuple[int, ...], ...],
    *,
    argnums: int | tuple[int, ...] = 0,
) -> SparsityPattern:
    """Detect global Jacobian sparsity pattern for f: R^n -> R^m.

    Analyzes the computation graph structure directly,
    without evaluating any derivatives.
    The result is valid for all inputs.

    Args:
        f: Function taking an array and returning an array.
        input_shape: Shape of the input array.
            An integer is treated as a 1D length.

    Returns:
        SparsityPattern of shape ``(m, n)``
            where ``n = prod(input_shape)`` and ``m = prod(output_shape)``.
            Entry ``(i, j)`` is present if output ``i`` depends on input ``j``.
    """
    argnums_tup = (argnums,) if isinstance(argnums, int) else tuple(argnums)

    if input_shape and isinstance(input_shape[0], int):
        shapes = (input_shape,)
    else:
        shapes = input_shape

    dummy_inputs = [jnp.zeros(s) for s in shapes]
    closed_jaxpr, out_struct = jax.make_jaxpr(f, return_shape=True)(*dummy_inputs)
    jaxpr = closed_jaxpr.jaxpr

    m = math.prod(out_struct.shape) if out_struct.shape else 1

    input_indices = []
    current_offset = 0

    for i, shape in enumerate(shapes):
        n = math.prod(shape) # We know it's a tuple now
        if i in argnums_tup:
            input_indices.append(offset_identity_index_sets(n, current_offset))
            current_offset += n
        else:
            input_indices.append(empty_index_sets(n))
            
    total_tracked_size = current_offset

    state_consts = {
        var: np.asarray(val)
        for var, val in zip(jaxpr.constvars, closed_jaxpr.consts, strict=False)
    }

    output_indices_list = prop_jaxpr(jaxpr, input_indices, state_consts)

    out_indices = []
    for out_deps in output_indices_list:
        out_indices.extend(out_deps)

    rows = []
    cols = []
    for i, deps in enumerate(out_indices):
        for j in deps:
            rows.append(i)
            cols.append(j)

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
