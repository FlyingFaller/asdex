"""Sparse Jacobian and Hessian computation using coloring and AD."""

from collections.abc import Callable
from typing import assert_never

import numpy as np
import math

import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from numpy.typing import ArrayLike

from asdex.coloring import hessian_coloring as _hessian_coloring
from asdex.coloring import jacobian_coloring as _jacobian_coloring
from asdex.detection import _ensure_scalar
from asdex.modes import (
    HessianMode,
    JacobianMode,
    _assert_hessian_mode,
    _assert_jacobian_mode,
)
from asdex.pattern import ColoredPattern

# Public API

def jacobian(
    f: Callable,
    input_shape: tuple[int, ...] | tuple[tuple[int, ...], ...],
    *,
    argnums: int | tuple[int, ...] = 0,
    mode: JacobianMode | None = None,
    symmetric: bool = False,
) -> Callable[..., BCOO | tuple[BCOO, ...]]:
    """Detect sparsity, color, and return a function computing sparse Jacobians.

    Combines [`jacobian_coloring`][asdex.jacobian_coloring]
    and [`jacobian_from_coloring`][asdex.jacobian_from_coloring]
    in one call.

    Args:
        f: Function taking an array and returning an array.
            Input and output may be multi-dimensional.
        input_shape: Shape of the input array.
        mode: AD mode.
            ``"fwd"`` uses JVPs (forward-mode AD),
            ``"rev"`` uses VJPs (reverse-mode AD).
            ``None`` picks whichever of fwd/rev needs fewer colors.
        symmetric: Whether to use symmetric (star) coloring.
            Requires a square Jacobian.

    Returns:
        A function that takes an input array and returns
            the sparse Jacobian as BCOO of shape ``(m, n)``
            where ``n = x.size`` and ``m = prod(output_shape)``.
    """
    coloring = _jacobian_coloring(f, input_shape, argnums=argnums, mode=mode, symmetric=symmetric)
    return jacobian_from_coloring(f, coloring, input_shape, argnums=argnums)


def value_and_jacobian(
    f: Callable[[ArrayLike], ArrayLike],
    input_shape: int | tuple[int, ...],
    *,
    mode: JacobianMode | None = None,
    symmetric: bool = False,
) -> Callable[[ArrayLike], tuple[jax.Array, BCOO]]:
    """Detect sparsity, color, and return a function computing value and sparse Jacobian.

    Like [`jacobian`][asdex.jacobian],
    but also returns the primal value ``f(x)``
    without an extra forward pass.

    Args:
        f: Function taking an array and returning an array.
            Input and output may be multi-dimensional.
        input_shape: Shape of the input array.
        mode: AD mode.
            ``"fwd"`` uses JVPs (forward-mode AD),
            ``"rev"`` uses VJPs (reverse-mode AD).
            ``None`` picks whichever of fwd/rev needs fewer colors.
        symmetric: Whether to use symmetric (star) coloring.
            Requires a square Jacobian.

    Returns:
        A function that takes an input array and returns
            ``(f(x), J)`` where ``J`` is the sparse Jacobian as BCOO
            of shape ``(m, n)``.
    """
    coloring = _jacobian_coloring(f, input_shape, mode=mode, symmetric=symmetric)
    return value_and_jacobian_from_coloring(f, coloring)


def hessian(
    f: Callable[[ArrayLike], ArrayLike],
    input_shape: int | tuple[int, ...],
    *,
    mode: HessianMode | None = None,
    symmetric: bool = True,
) -> Callable[[ArrayLike], BCOO]:
    """Detect sparsity, color, and return a function computing sparse Hessians.

    Combines [`hessian_coloring`][asdex.hessian_coloring]
    and [`hessian_from_coloring`][asdex.hessian_from_coloring]
    in one call.

    If ``f`` returns a squeezable shape like ``(1,)`` or ``(1, 1)``,
    it is automatically squeezed to scalar.

    Args:
        f: Scalar-valued function taking an array.
            Input may be multi-dimensional.
        input_shape: Shape of the input array.
        mode: AD composition strategy for Hessian-vector products.
            ``"fwd_over_rev"`` uses forward-over-reverse,
            ``"rev_over_fwd"`` uses reverse-over-forward,
            ``"rev_over_rev"`` uses reverse-over-reverse.
            Defaults to ``"fwd_over_rev"``.
        symmetric: Whether to use symmetric (star) coloring.
            Defaults to True (exploits H = H^T for fewer colors).

    Returns:
        A function that takes an input array and returns
            the sparse Hessian as BCOO of shape ``(n, n)``
            where ``n = x.size``.
    """
    coloring = _hessian_coloring(f, input_shape, mode=mode, symmetric=symmetric)
    return hessian_from_coloring(f, coloring)


def value_and_hessian(
    f: Callable[[ArrayLike], ArrayLike],
    input_shape: int | tuple[int, ...],
    *,
    mode: HessianMode | None = None,
    symmetric: bool = True,
) -> Callable[[ArrayLike], tuple[jax.Array, BCOO]]:
    """Detect sparsity, color, and return a function computing value and sparse Hessian.

    Like [`hessian`][asdex.hessian],
    but can also return the primal value ``f(x)``
    without an extra forward pass.

    If ``f`` returns a squeezable shape like ``(1,)`` or ``(1, 1)``,
    it is automatically squeezed to scalar.

    Args:
        f: Scalar-valued function taking an array.
            Input may be multi-dimensional.
        input_shape: Shape of the input array.
        mode: AD composition strategy for Hessian-vector products.
            ``"fwd_over_rev"`` uses forward-over-reverse,
            ``"rev_over_fwd"`` uses reverse-over-forward,
            ``"rev_over_rev"`` uses reverse-over-reverse.
            Defaults to ``"fwd_over_rev"``.
        symmetric: Whether to use symmetric (star) coloring.
            Defaults to True (exploits H = H^T for fewer colors).

    Returns:
        A function that takes an input array and returns
            ``(f(x), H)`` where ``H`` is the sparse Hessian as BCOO
            of shape ``(n, n)``.
    """
    coloring = _hessian_coloring(f, input_shape, mode=mode, symmetric=symmetric)
    return value_and_hessian_from_coloring(f, coloring)


def jacobian_from_coloring(
    f: Callable,
    coloring: ColoredPattern,
    input_shape: tuple[int, ...] | tuple[tuple[int, ...], ...],
    *, 
    argnums: int | tuple[int, ...] = 0,
) -> Callable[..., BCOO | tuple[BCOO, ...]]:
    """Build a sparse Jacobian function from a pre-computed coloring.

    Uses row coloring + VJPs or column coloring + JVPs,
    depending on which needs fewer colors.

    Args:
        f: Function taking an array and returning an array.
            Input and output may be multi-dimensional.
        coloring: Pre-computed [`ColoredPattern`][asdex.ColoredPattern]
            from [`jacobian_coloring`][asdex.jacobian_coloring].

    Returns:
        A function that takes an input array and returns
            the sparse Jacobian as BCOO of shape ``(m, n)``
            where ``n = x.size`` and ``m = prod(output_shape)``.
    """

    # Force argnums to be a tuple iterable
    argnums_tup = (argnums,) if isinstance(argnums, int) else tuple(argnums)

    # Make shapes a tuple of shape tuples
    if input_shape and isinstance(input_shape[0], int):
        shapes = (input_shape,)
    else:
        shapes = input_shape    

    # Setup split
    target_specs = []
    current_offset = 0

    color_idx_full, elem_idx_full = coloring._extraction_indices
    data_cols = np.asarray(coloring.sparsity.cols)
    data_rows = np.asarray(coloring.sparsity.rows)

    # Precompute split indices
    for i, shape in enumerate(shapes):
        size = math.prod(shape)
        if i in argnums_tup:
            mask = (data_cols >= current_offset) & (data_cols < current_offset + size)

            sub_color_idx = color_idx_full[mask]
            sub_elem_idx = elem_idx_full[mask]
            
            sub_rows = data_rows[mask]
            sub_cols = data_cols[mask] - current_offset
            sub_indices = np.column_stack((sub_rows, sub_cols))
            
            target_specs.append({
                "size": size,
                "color_idx": jnp.asarray(sub_color_idx),
                "elem_idx": jnp.asarray(sub_elem_idx),
                "indices": jnp.asarray(sub_indices, dtype=jnp.int32)
            })
            current_offset += size

    # Create function
    def jac_fn(*args: ArrayLike):
        args_jnp = tuple(jnp.asarray(x) for x in args)
        return _eval_jacobian(f, args_jnp, argnums_tup, coloring, target_specs)

    return jac_fn


def hessian_from_coloring(
    f: Callable[[ArrayLike], ArrayLike],
    coloring: ColoredPattern,
) -> Callable[[ArrayLike], BCOO]:
    """Build a sparse Hessian function from a pre-computed coloring.

    Uses symmetric (star) coloring and Hessian-vector products by default.

    If ``f`` returns a squeezable shape like ``(1,)`` or ``(1, 1)``,
    it is automatically squeezed to scalar.

    Args:
        f: Scalar-valued function taking an array.
            Input may be multi-dimensional.
        coloring: Pre-computed [`ColoredPattern`][asdex.ColoredPattern]
            from [`hessian_coloring`][asdex.hessian_coloring].

    Returns:
        A function that takes an input array and returns
            the sparse Hessian as BCOO of shape ``(n, n)``
            where ``n = x.size``.
    """

    def hess_fn(x: ArrayLike) -> BCOO:
        return _eval_hessian(f, jnp.asarray(x), coloring)

    return hess_fn


def value_and_jacobian_from_coloring(
    f: Callable[[ArrayLike], ArrayLike],
    coloring: ColoredPattern,
) -> Callable[[ArrayLike], tuple[jax.Array, BCOO]]:
    """Build a function computing value and sparse Jacobian from a pre-computed coloring.

    Like [`jacobian_from_coloring`][asdex.jacobian_from_coloring],
    but also returns the primal value ``f(x)`` without an extra forward pass.

    Args:
        f: Function taking an array and returning an array.
            Input and output may be multi-dimensional.
        coloring: Pre-computed [`ColoredPattern`][asdex.ColoredPattern]
            from [`jacobian_coloring`][asdex.jacobian_coloring].

    Returns:
        A function that takes an input array and returns
            ``(f(x), J)`` where ``J`` is the sparse Jacobian as BCOO
            of shape ``(m, n)``.
    """

    def val_jac_fn(x: ArrayLike) -> tuple[jax.Array, BCOO]:
        return _eval_value_and_jacobian(f, jnp.asarray(x), coloring)

    return val_jac_fn


def value_and_hessian_from_coloring(
    f: Callable[[ArrayLike], ArrayLike],
    coloring: ColoredPattern,
) -> Callable[[ArrayLike], tuple[jax.Array, BCOO]]:
    """Build a function computing value and sparse Hessian from a pre-computed coloring.

    Like [`hessian_from_coloring`][asdex.hessian_from_coloring],
    but can also return the primal value ``f(x)`` without an extra forward pass.

    If ``f`` returns a squeezable shape like ``(1,)`` or ``(1, 1)``,
    it is automatically squeezed to scalar.

    Args:
        f: Scalar-valued function taking an array.
            Input may be multi-dimensional.
        coloring: Pre-computed [`ColoredPattern`][asdex.ColoredPattern]
            from [`hessian_coloring`][asdex.hessian_coloring].

    Returns:
        A function that takes an input array and returns
            ``(f(x), H)`` where ``H`` is the sparse Hessian as BCOO
            of shape ``(n, n)``.
    """

    def val_hess_fn(x: ArrayLike) -> tuple[jax.Array, BCOO]:
        return _eval_value_and_hessian(f, jnp.asarray(x), coloring)

    return val_hess_fn


# Internal evaluation logic


def _eval_jacobian(
    f: Callable,
    args: tuple[jax.Array, ...],
    argnums_tup: tuple[int, ...],
    coloring: ColoredPattern,
    target_specs: list[dict],
) -> BCOO | tuple[BCOO, ...]:
    """Evaluate the sparse Jacobian of f at x."""
    m = coloring.sparsity.m

    # Handle edge cases: no outputs or all-zero Jacobian
    if m == 0 or coloring.sparsity.nnz == 0:
        results = [
            BCOO((jnp.array([]), jnp.zeros((0, 2), dtype=jnp.int32)), shape=(m, spec["size"]))
            for spec in target_specs
        ]
        return results[0] if len(target_specs) == 1 else tuple(results)

    # Do computation using correct mode
    _assert_jacobian_mode(coloring.mode)
    match coloring.mode:
        case "rev": 
            compressed = _jacobian_rows(f, args, argnums_tup, coloring)
        case "fwd":
            compressed = _jacobian_cols(f, args, argnums_tup, coloring)
        case _ as unreachable:
            assert_never(unreachable)

    # Decompress results for multi-arg
    results = []
    for spec in target_specs:
        sub_data = compressed[spec["color_idx"], spec["elem_idx"]]
        bcoo = BCOO((sub_data, spec["indices"]), shape=(m, spec["size"]))
        results.append(bcoo)

    return results[0] if len(results) == 1 else tuple(results)


def _eval_hessian(
    f: Callable[[ArrayLike], ArrayLike],
    x: jax.Array,
    coloring: ColoredPattern,
) -> BCOO:
    """Evaluate the sparse Hessian of f at x.

    If ``f`` returns a squeezable shape like ``(1,)``,
    it is automatically squeezed to scalar.
    """
    f = _ensure_scalar(f, x.shape)
    n = x.size

    expected = coloring.sparsity.input_shape
    if x.shape != expected:
        raise ValueError(
            f"Input shape {x.shape} does not match the colored pattern, "
            f"which expects shape {expected}."
        )

    sparsity = coloring.sparsity

    # Handle edge case: all-zero Hessian
    if sparsity.nnz == 0:
        return BCOO((jnp.array([]), jnp.zeros((0, 2), dtype=jnp.int32)), shape=(n, n))

    grads = _compute_hvps(f, x, coloring)
    return _decompress(coloring, grads)


def _eval_value_and_jacobian(
    f: Callable[[ArrayLike], ArrayLike],
    x: jax.Array,
    coloring: ColoredPattern,
) -> tuple[jax.Array, BCOO]:
    """Evaluate f(x) and the sparse Jacobian of f at x."""
    n = x.size

    expected = coloring.sparsity.input_shape
    if x.shape != expected:
        raise ValueError(
            f"Input shape {x.shape} does not match the colored pattern, "
            f"which expects shape {expected}."
        )

    sparsity = coloring.sparsity
    m = sparsity.m
    out_shape = jax.eval_shape(f, jnp.zeros_like(x)).shape

    # Handle edge case: no outputs
    if m == 0:
        y = jnp.asarray(f(x))
        return y, BCOO(
            (jnp.array([]), jnp.zeros((0, 2), dtype=jnp.int32)), shape=(0, n)
        )

    # Handle edge case: all-zero Jacobian
    if sparsity.nnz == 0:
        y = jnp.asarray(f(x))
        return y, BCOO(
            (jnp.array([]), jnp.zeros((0, 2), dtype=jnp.int32)), shape=(m, n)
        )

    _assert_jacobian_mode(coloring.mode)
    match coloring.mode:
        case "rev":
            return _value_and_jacobian_rows(f, x, coloring, out_shape)
        case "fwd":
            return _value_and_jacobian_cols(f, x, coloring)
        case _ as unreachable:
            assert_never(unreachable)  # type: ignore[type-assertion-failure]


def _eval_value_and_hessian(
    f: Callable[[ArrayLike], ArrayLike],
    x: jax.Array,
    coloring: ColoredPattern,
) -> tuple[jax.Array, BCOO]:
    """Evaluate f(x) and the sparse Hessian of f at x.

    If ``f`` returns a squeezable shape like ``(1,)``,
    it is automatically squeezed to scalar.
    """
    f = _ensure_scalar(f, x.shape)
    n = x.size

    expected = coloring.sparsity.input_shape
    if x.shape != expected:
        raise ValueError(
            f"Input shape {x.shape} does not match the colored pattern, "
            f"which expects shape {expected}."
        )

    sparsity = coloring.sparsity

    # Handle edge case: all-zero Hessian
    if sparsity.nnz == 0:
        y = jnp.asarray(f(x))
        return y, BCOO(
            (jnp.array([]), jnp.zeros((0, 2), dtype=jnp.int32)), shape=(n, n)
        )

    value, grads = _value_and_compute_hvps(f, x, coloring)
    return value, _decompress(coloring, grads)


# Private helpers: Jacobian


def _jacobian_rows(
    f: Callable,
    args: tuple[jax.Array, ...],
    argnums_tup: tuple[int, ...],
    coloring: ColoredPattern,
) -> jax.Array:
    """Compute sparse Jacobian via row coloring + VJPs."""

    seeds = jnp.asarray(coloring._seed_matrix, dtype=args[0].dtype)
    _, vjp_fn = jax.vjp(f, *args)
    
    out_struct = jax.eval_shape(f, *args)
    out_leaves, out_treedef = jax.tree_util.tree_flatten(out_struct)

    def single_vjp(seed: jax.Array) -> jax.Array:
        seed_leaves = []
        offset = 0
        for leaf in out_leaves:
            size = math.prod(leaf.shape) if leaf.shape else 1
            seed_leaves.append(seed[offset : offset + size].reshape(leaf.shape))
            offset += size

        seed_unflat = jax.tree_util.tree_unflatten(out_treedef, seed_leaves)
        vjp_outs = vjp_fn(seed_unflat)

        grads = []
        for i in argnums_tup:
            if vjp_outs[i] is not None:
                grads.append(jnp.atleast_1d(vjp_outs[i]).ravel())
            else:
                grads.append(jnp.zeros(args[i].size))
        return jnp.concatenate(grads)

    return jax.vmap(single_vjp)(seeds)

def _value_and_jacobian_rows(
    f: Callable[[ArrayLike], ArrayLike],
    x: jax.Array,
    coloring: ColoredPattern,
    out_shape: tuple[int, ...],
) -> tuple[jax.Array, BCOO]:
    """Compute value and sparse Jacobian via row coloring + VJPs.

    The primal is free from the VJP forward pass.
    """
    seeds = jnp.asarray(coloring._seed_matrix, dtype=x.dtype)
    y, vjp_fn = jax.vjp(f, x)

    def single_vjp(seed: jax.Array) -> jax.Array:
        (grad,) = vjp_fn(seed.reshape(out_shape))
        return grad.ravel()

    return y, _decompress(coloring, jax.vmap(single_vjp)(seeds))


def _jacobian_cols(
    f: Callable,
    args: tuple[jax.Array, ...],
    argnums_tup: tuple[int, ...],
    coloring: ColoredPattern,
) -> jax.Array:
    seeds = jnp.asarray(coloring._seed_matrix, dtype=args[0].dtype)

    def single_jvp(seed: jax.Array) -> jax.Array:
        tangents = list(jnp.zeros_like(x) for x in args)
        offset = 0
        for i in argnums_tup:
            size = args[i].size
            tangents[i] = seed[offset : offset + size].reshape(args[i].shape)
            offset += size

        _, jvp_out = jax.jvp(f, args, tuple(tangents))
        out_leaves = jax.tree_util.tree_leaves(jvp_out)
        return jnp.concatenate([jnp.atleast_1d(o).ravel() for o in out_leaves])

    return jax.vmap(single_jvp)(seeds)


def _value_and_jacobian_cols(
    f: Callable[[ArrayLike], ArrayLike],
    x: jax.Array,
    coloring: ColoredPattern,
) -> tuple[jax.Array, BCOO]:
    """Compute value and sparse Jacobian via column coloring + JVPs.

    Uses ``jax.linearize`` so the nonlinear forward pass runs only once.
    """
    seeds = jnp.asarray(coloring._seed_matrix, dtype=x.dtype)

    y, jvp_fn = jax.linearize(f, x)

    def single_jvp(seed: jax.Array) -> jax.Array:
        return jvp_fn(seed.reshape(x.shape)).ravel()

    tangents = jax.vmap(single_jvp)(seeds)
    return y, _decompress(coloring, tangents)


# Private helpers: Hessian


def _compute_hvps(
    f: Callable[[ArrayLike], ArrayLike],
    x: jax.Array,
    coloring: ColoredPattern,
) -> jax.Array:
    """Compute one HVP per color using pre-computed seed matrix.

    Returns ``hvps`` of shape ``(num_colors, n)``.
    """
    seeds = jnp.asarray(coloring._seed_matrix, dtype=x.dtype)

    _assert_hessian_mode(coloring.mode)
    match coloring.mode:
        case "fwd_over_rev":
            _, hvp_fn = jax.linearize(jax.grad(f), x)

            def single_hvp(v: jax.Array) -> jax.Array:
                return hvp_fn(v.reshape(x.shape)).ravel()

        case "rev_over_fwd":

            def single_hvp(v: jax.Array) -> jax.Array:
                return jax.grad(lambda p: jax.jvp(f, (p,), (v.reshape(x.shape),))[1])(
                    x
                ).ravel()

        case "rev_over_rev":
            _, hvp_fn = jax.vjp(jax.grad(f), x)

            def single_hvp(v: jax.Array) -> jax.Array:
                (hvp,) = hvp_fn(v.reshape(x.shape))
                return hvp.ravel()

        case _ as unreachable:
            assert_never(unreachable)  # type: ignore[type-assertion-failure]

    return jax.vmap(single_hvp)(seeds)


def _value_and_compute_hvps(
    f: Callable[[ArrayLike], ArrayLike],
    x: jax.Array,
    coloring: ColoredPattern,
) -> tuple[jax.Array, jax.Array]:
    """Compute ``f(x)`` and one HVP per color using pre-computed seed matrix.

    Returns ``(f(x), hvps)`` where ``hvps`` has shape ``(num_colors, n)``.
    The primal is free for ``fwd_over_rev`` and ``rev_over_rev``;
    ``rev_over_fwd`` computes it with a separate ``f(x)`` call.
    """
    seeds = jnp.asarray(coloring._seed_matrix, dtype=x.dtype)

    _assert_hessian_mode(coloring.mode)
    match coloring.mode:
        case "fwd_over_rev":
            (value, _grad_at_x), hvp_fn = jax.linearize(jax.value_and_grad(f), x)

            def single_hvp(v: jax.Array) -> jax.Array:
                _tangent_of_value, hvp = hvp_fn(v.reshape(x.shape))
                return hvp.ravel()

        case "rev_over_fwd":
            value = jnp.asarray(f(x))

            def single_hvp(v: jax.Array) -> jax.Array:
                return jax.grad(lambda p: jax.jvp(f, (p,), (v.reshape(x.shape),))[1])(
                    x
                ).ravel()

        case "rev_over_rev":
            # TODO: f(x) is redundant with the forward pass inside grad(f).
            # Using value_and_grad + vjp would avoid it, but inflates every
            # VJP application with dead zero-cotangents for the value path.
            # Revisit if XLA reliably DCEs the zero branch.
            value = jnp.asarray(f(x))
            _, hvp_fn = jax.vjp(jax.grad(f), x)

            def single_hvp(v: jax.Array) -> jax.Array:
                (hvp,) = hvp_fn(v.reshape(x.shape))
                return hvp.ravel()

        case _ as unreachable:
            assert_never(unreachable)  # type: ignore[type-assertion-failure]

    return value, jax.vmap(single_hvp)(seeds)


# Private helpers: decompression


def _decompress(coloring: ColoredPattern, compressed: jax.Array) -> BCOO:
    """Extract sparse entries from compressed gradient rows.

    Uses pre-computed extraction indices on the ``ColoredPattern``
    to vectorize the decompression step
    (no Python loop over nnz entries).

    Args:
        coloring: Colored sparsity pattern with cached indices.
        compressed: JAX array of shape (num_colors, vector_len),
            one row per color.

    Returns:
        Sparse matrix as BCOO in sparsity-pattern order.
    """
    color_idx, elem_idx = coloring._extraction_indices
    data = compressed[jnp.asarray(color_idx), jnp.asarray(elem_idx)]
    return coloring.sparsity.to_bcoo(data=data)
