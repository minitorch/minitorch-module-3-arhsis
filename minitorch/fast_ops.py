from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_size = int(np.prod(out_shape))
        # out_idx is a shape x out_size array
        # in_idx is a shape x in_size array
        out_idx_size = len(out_shape) * out_size
        out_idx_shape = (out_size, len(out_shape))
        out_idx = np.zeros(out_idx_size, dtype=np.int32).reshape(out_idx_shape)

        in_idx_size = len(in_shape) * out_size
        in_idx_shape = (out_size, len(in_shape))
        in_idx = np.zeros(in_idx_size, dtype=np.int32).reshape(in_idx_shape)

        for out_pos in prange(out_size):
            to_index(out_pos, out_shape, out_idx[out_pos])
            # out_shape is broadcasted from in_shape, mapping out_idx to in_idx
            broadcast_index(out_idx[out_pos], out_shape,
                            in_shape, in_idx[out_pos])
            # get the in pos by in_idx
            in_pos = index_to_position(in_idx[out_pos], in_strides)
            out[out_pos] = fn(in_storage[in_pos])

    return njit(_map, parallel=True)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # mapping out_pos to a_pos and b_pos
        out_size = int(np.prod(out_shape))
        out_idx = [0] * len(out_shape)
        a_idx = [0] * len(a_shape)
        b_idx = [0] * len(b_shape)

        out_idx_size = len(out_shape) * out_size
        out_idx_shape = (out_size, len(out_shape))
        out_idxes = np.zeros(out_idx_size, dtype=np.int32).reshape(out_idx_shape)

        a_idx_size = len(a_shape) * out_size
        a_idx_shape = (out_size, len(a_shape))
        a_idxes = np.zeros(a_idx_size, dtype=np.int32).reshape(a_idx_shape)

        b_idx_size = len(b_shape) * out_size
        b_idx_shape = (out_size, len(b_shape))
        b_idxes = np.zeros(b_idx_size, dtype=np.int32).reshape(b_idx_shape)

        for out_pos in prange(out_size):
            out_idx = out_idxes[out_pos]
            a_idx, b_idx = a_idxes[out_pos], b_idxes[out_pos]
            to_index(out_pos, out_shape, out_idx)
            broadcast_index(out_idx, out_shape,
                            a_shape, a_idx)
            broadcast_index(out_idx, out_shape,
                            b_shape, b_idx)
            # get the a pos and b pos by a_idx and b_idx
            a_pos = index_to_position(a_idx, a_strides)
            b_pos = index_to_position(b_idx, b_strides)
            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return njit(_zip, parallel=True)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
    ----
        fn: reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        def reduce_over_dimension(a_idx: np.ndarray, out_pos: int):
            """
            Reduce over the dimension `reduce_dim` in `a`
            """
            for i in range(a_shape[reduce_dim]):
                a_idx[reduce_dim] = i
                a_pos = index_to_position(a_idx, a_strides)
                out[out_pos] = fn(out[out_pos], a_storage[a_pos])

        out_size = int(np.prod(out_shape))
        out_idx_size = len(out_shape) * out_size
        out_idx_shape = (out_size, len(out_shape))
        out_idx = np.zeros(out_idx_size, dtype=np.int32).reshape(out_idx_shape)

        a_idx_size = len(a_shape) * out_size
        a_idx_shape = (out_size, len(a_shape))
        a_idx = np.zeros(a_idx_size, dtype=np.int32).reshape(a_idx_shape)

        for out_pos in prange(out_size):
            to_index(out_pos, out_shape, out_idx[out_pos])
            # mapping out_idx to a_idx
            broadcast_index(out_idx[out_pos], out_shape,
                            a_shape, a_idx[out_pos])
            # reduce the reduce_dim
            reduce_over_dimension(a_idx[out_pos], out_pos)

    return njit(_reduce, parallel=True)  # type: ignore


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
    -------
        None : Fills in `out`

    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    out_batch_stride = out_strides[0] if out_shape[0] > 1 else 0

    # Create batch position arrays using list comprehension with ternary operators
    a_batch_pos = [0] * out_shape[0] if a_shape[0] == 1 else [a_batch_stride * i for i in range(a_shape[0])]
    b_batch_pos = [0] * out_shape[0] if b_shape[0] == 1 else [b_batch_stride * i for i in range(b_shape[0])]
    out_batch_pos = [out_batch_stride * i for i in range(out_shape[0])]


    for b in prange(out_shape[0]):
        for i in range(out_shape[1]):
            for j in range(out_shape[2]):
                # find out the position relationship of out, a and b
                out_pos = out_batch_pos[b] + out_strides[1] * i + out_strides[2] * j
                a_pos_start = a_batch_pos[b] + a_strides[1] * i
                b_pos_start = b_batch_pos[b] + b_strides[2] * j
                for k in range(a_shape[-1]):
                    a_pos = a_pos_start + a_strides[-1] * k
                    b_pos = b_pos_start + b_strides[1] * k
                    out[out_pos] += a_storage[a_pos] * b_storage[b_pos]



tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
