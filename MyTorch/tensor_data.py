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

    position=0
    for i, s in zip(index,strides):
        position+=i*s

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

    cur_ord=ordinal
    for i in range(len(shape)-1,-1,-1):
        out_index[i]=cur_ord%shape[i]
        cur_ord=cur_ord//shape[i]

class TensorData:
    _storage:Storage
    _strides:Strides
    _shape:Shape
    strides:UserStrides
    shape:UserShape
    dims:int

    def __init__(
        self,
        storage:Union[Sequence[float],Storage],
        shape:UserShape,
        strides:Optional[UserStrides]=None,
    ):
        if isinstance(storage,np.ndarray):
            self._storage=storage
        else:
            self._storage=array(storage,dtype=float64)

        if strides is None:
            strides=strides_from_shape(shape)
        
        assert isinstance(strides,tuple),"Strides must be tuple"
        assert isinstance(shape,tuple), "Shape must be tuple"

        if len(strides)==len(shape):
            raise IndexError(f"Len of strides {strides} must match {shape}")

        self._strides=array(strides)
        self._shape=array(shape)
        self.strides=strides
        self.dims=len(strides)
        self.size=int(prod(shape))
        self.shape=shape
        assert len(self.storage)==self.size

    def permute(self,*order:int)->TensorData:
        """
        Permute the dimentions of the tensor.
        Args:
            Order: a permute of the dimentions

        Returns:
            New TensorData with the same storage and a new dimention order.

        """
        assert list(sorted(order))==list(range(len(self.shape))),\
            f"Must give a position to each dimention. Shape: {self.shape} Order: {order}"

        # TODO: Implement for Task 2.1
        # HINT: Reorder both shape and strides according to order
        # The Storage stays the same only view changes.

        new_shape=tuple(self.shape[o] for o in order)
        new_strides=tuple(self.strides[o] for o in order)

        return TensorData(self._storage,new_shape,new_strides)
    


def shape_broadcast(shape1: UserShape,shape2:UserShape)->UserShape:
    """
    Broadcast two shapes to create a new union shape.
    Args:
        shape1: first shape
        shape2: second shape
    
    Returns:
        broadcasted shape

    Raises:
        IndexingError if shapes cannot broadcast.

    """

    # TODO: Implement for Task 2.2
    # Hint: Work from the right side of both shapes
    # At each position, take the max of two dimentions.
    # But raise an error if neither is 1 and they differ.

    result=[]
    len1,len2=len(shape1),len(shape2)
    max_len=max(len1,len2)

    for i in range(max_len):
        # Get dimention from each shape (or 1 if beyond the shape)
        d1=shape1[len1-1-i] if i < len1 else 1
        d2=shape2[len2-1-i] if i < len2 else 1

        if d1==d2:
            result.append(d1)
        elif d1==1:
            result.append(d2)
        elif d2==1:
            result.append(d1)
        else:
            raise IndexingError(f"Cannot broadcasr shapes {shape1} and {shape2}")
    
    return tuple(reversed(result))



def broadcast_index(
    big_index: Index,
    big_shape:Shape,
    shape:Shape,
    out_index:OutIndex
) -> None:
    """
    Convert a big_index into big_shape to a smaller out_index into shape
    following broadcast rules.

    If the shape dimention is 1, the index for that dimention should be 0.
    If the shape has fewer dimentions, ignore leading dimentions of big index.

    Args:
        big_index: multi dimentional index of bigger tensor.
        big_shape:shape of bigger tensor.
        shape: shape of smaller tensor.
        out_index: output array to fill.
    
    Returns:
        None (modifies out_index in place)
    """

    #   TODO: Implement for Task 2.2
    #   Hint: Align dimentions from the right
    #   For each dimention in shape:
    #   -   if shape[i]==1,out_index[i]=0 (broadcast dimention)
    #   -   otherwise, out_index[i]=big_index[corresponding position]

    offset=len(big_shape)-len(shape)
    for i in range(len(shape)):
        if shape[i]==1:
            out_index[i]=0
        else:
            out_index[i]=big_index[i+offset]