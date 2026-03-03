"""
autho diff utility for minitorch.

"""

from typing import Callable,List,Tuple,Any
from dataclasses import dataclass
from typing import Optional,Sequence
from typing import List,Set
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from MyTorch.scalar import Context


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
    history: Optional["History"]=None # last_fn,ctx,inputs
    derivative:Optional[float]=None # store derivative during backprop
    name: Optional[str]=None # Storing name for debugging,
    
    @property
    def is_leaf(self)->bool:
        """A leaf variable has no history (was not created by an operation)."""
        return self.history is None
    
    @property
    def is_constant(self)->bool:
        """A constant has no history and will not receive gradients."""
        return self.history is None
    
    def requires_grad_(self,requires_grad:bool=True)->"Variable":
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
    last_fn:Optional[type]=None # What function was used.
    ctx:Optional["Context"]=None   # variables that we will need during computing backprop
    inputs: Sequence["Variable"]=() # which inputs was used
    
def topological_sort(variable:Variable)->List[Variable]:

    """
    Return variables in topological order (children before parents).
    For backpropagration, we need to process a  variable after all,
    varaible that depend on it have been processed.
    """

    order: List[Variable]=[]
    visited:Set[int]=set()

    def visit(var:Variable)->None:
        var_id=id(var)

        if var_id in visited:
            return

        visited.add(var_id)

        # Visit children first (variables this one depends on)
        if var.history is not None and var.history.inputs:
            for input_var in var.history.inputs:
                visit(input_var)

        order.append(var)
    # Starting dfs+topological sort.
    visit(variable)

    return order


def backpropagate(variable:Variable,deriv:float=1.0)->None:
    """
    Run backpropagration start from variables
    Computes gradients for all leaf variables in the computation graph.

    Args: 
        variables: Output variable to differntiate (e.g., loss)
        deriv: Graident of variable (default 1.0 for scalar loss)
    """

    # Get variables in topological order.
    sorted_vars=topological_sort(variable)

    # Process in Reverse topological order
    sorted_vars=list(reversed(sorted_vars))

    #Initialize gradient of output
    variable.derivative=deriv
    
    for var in sorted_vars:
        if var.is_leaf:
            # leaf variables just acculomate gradients,
            #  nothing to propagate.
            continue
        if var.derivative is None:
            # No gradient reached this node (disconnected).
            continue

        history:"History"=var.history
        if history is None or history.last_fn is None:
            continue

        # Call backward to get gradient for the inputs.
        backward_fn=history.last_fn.backward
        ctx=history.ctx
        input_grads=backward_fn(ctx,var.derivative)

        #  Acculomate gradients to input variables.
        for input_var,grad in zip(history.inputs,input_grads):
            if grad is not None:
                input_var.accumulate_derivative(grad)








    