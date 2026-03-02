"""
autho diff utility for minitorch.

"""

from typing import Callable,List,Tuple,Any
from dataclasses import dataclass
from typing import Optional,Sequence
from typing import List,Set
def central_difference(f:Callable[...,float],*vals:float,arg:int=0,epsilon:float=1e-6)->float:
    """
    Compute numerical derivative of f with respect to argument `arg`.

    Uses central difference: (f(x+h) - f(x-h)) / (2h)

    Args:
        f: Function to differentiate
        *vals: Input values to f
        arg: Which argument to differentiate with respect to (0-indexed)
        epsilon: Step size for numerical differentiation

    Returns:
        Approximate derivative

    Example:
        >>> def mul(x, y): return x * y
        >>> central_difference(mul, 3, 4, arg=0)  # df/dx at (3,4)
        4.0
        >>> central_difference(mul, 3, 4, arg=1)  # df/dy at (3,4)
        3.0
    """
    vals_list=list(vals)
    # Create vals with arg incremented by epsilon.
    vals_plus=vals_list.copy()
    vals_plus[arg]=vals_plus[arg]+epsilon

    # Create vals with arg decremented by epsilon
    vals_minus=vals_list.copy()
    vals_minus[arg]=vals_minus[arg]-epsilon

    # Compute central difference
    f_plus=f(*vals_plus)
    f_minus=f(*vals_minus)

    return (f_plus-f_minus)/(2*epsilon)


@dataclass
class Variable:
    """
    A node in the computation graph.
    Scalar,Tensor inherits from this
    Attributes:
        history:    Record of the operation that created this variable.
        derivative: Accumulated gradient (set during backward pass)
        name:   Optional name for debugging.
    """
    history: Optional["History"]=None
    derivative:Optional[float]=None
    name: Optional[str]=None

    def is_leaf(self)->bool:
        """A leaf variable has no history (was not created by an operation)."""
        return self.history is None
    
    def is_constant(self)->bool:
        """A constant has no history and will not receive gradients."""
        return self.history is None
    
    def requires_grad(self,requires_grad:bool=True)->"Variable":
        if requires_grad:
            self.history=History()
        else:
            self.history=None
        return self
    
@dataclass
class History:
    """
    Records the operation that created a variable.

    Attributes:
        last_fn: The function class that created this variable
       ctx: Context object storing values needed for backward
        inputs: The input variable to the operation.
    """
    last_fn:Optional[type]=None
    ctx:Optional["Context"]=None
    inputs: Sequence["Variable"]=()
    

def topological_sort(variable:Variable)->List[Variable]:

    """
    Return variables in topological order (children before parents).
    """
    