from __future__ import annotations
from typing import Iterable,Optional,Sequence,Tuple,Union
import numpy as np
import numpy.typing as npt

from numpy import array,float64

MAX_DIMS=32 

class IndexingError(RuntimeError):
    pass 

Storage=npt.NDArray[np.float64]
OutIndex=npt.NDArray[np.int32]
Index=npt.NDArray[np.int32]
Shape=npt.NDArray[np.int32]
Strides=npt.NDArray[np.int32]
UserIndex=Sequence[int]
UserShape=Sequence[int]
UserStrides=Sequence[int]


def index_to_position(index:Index,strides:Strides)->int:
    """"
    Convert a multidimentional tensor index into a single-dimentional
    position in storage based on strides

    Args:
        index: index tuple based of ints( as numpy array)
        strides: tensor strides ( as numpy as array)
    
    Returns:
        Position in storage
    """
    # TODO storage for task 2.1
    # Hint: The Position is the dot product of index and strides
    # Example: index[1,2], strides=[4,1]-> Strides=[4,1] -> position=1*4+2*1=6

    position=index*strides

    return position

def to_index(ordinal:int,shape:Shape,out_index:OutIndex):
    """
    Convert an ordinal (flat position 0....size-1) to a multi dimentional 
    index in the given shape.
    This is the inverse mapping: given position in enumeration order, 
    produce the corresponding index.

    Args:
        ordianal: ordinal position to covert
        shape: tensor shape
        out_index:  output array to fill with index values.
    
    Returns:
        None (modifies out index in place)
    """
    # TODO: Implement for task 2.1
    # Hint: Work from the last dimention to the first.
    # Use modulo and integer divison
    # Example: ordinal=5, shape=(2,3) -> out_index=[1,2]
    # 5%3=2 (last index)
    # 5//3 = 1 (remaining ordinal for next dimention)

