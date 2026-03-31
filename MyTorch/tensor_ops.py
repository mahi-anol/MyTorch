from __future__ import annotations
from typing import Callable,Optional,Type
import numpy as np

from MyTorch.tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
    Index,Shape,Storage,Strides,
)


def tensor_map(
        fn:Callable[[float],float]
) -> Callable[[Storage,Shape,Strides,Storage,Shape,Strides],None]:
    """
    Low-level implementation of tensor map betweem tensors with 
    possibly different strides.

    Args:
        fn: function from float to apply.
    
    Returns:
        Tensor map function.
    """

    def _map(
        out: Storage,
        out_shape:Shape,
        out_strides:Strides,
        in_storage:Storage,
        in_shape: Shape,
        in_strides:Strides
    ) -> None:
        # TODO: Implement for task 2.3
        # For each postion in output:
        # 1. Convert postion to output index 
        # 2. Map output index to input index (handle broadcasting)
        # 3. Get input value,apply fn, store in output

        out_index=np.zeros (MAX_DIMS,dtype=np.int32)
        in_index=np.zeros(MAX_DIMS,dtype=np.int32)

        for i in range(int(np.prod(out_Shape))):
            # Step 1: Get output index
            to_index(i,out_shape,out_index)

            # Step 2: Map to input index( handles broadcasting)
            broadcast_index(out_index,out_shape,in_shape,in_index)


            out_pos=index_to_position(out_index,out_strides)
            in_pos=index_to_position(in_index,in_strides)

            out[out_pos]=fn(in_pos)

    return _map

