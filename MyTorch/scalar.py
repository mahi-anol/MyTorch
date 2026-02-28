"""
Scalar class for auto differentiation on sigle values.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional,Sequence,Tuple,Type,Union

from MyTorch.autodiff import Variable,History,Context

@dataclass 
class ScalarHistory:
    """
    Records the operation that created a scalar.
    """
    last_fn:Optional[Type["ScalarFunction"]]=None
    ctx: Optional[Context]=None
    inputs=Sequence["Scalar"]=()


class Scalar(Variable):
    """
        A scalar value that tracks computation history for autodiff.

        Attributes:
            data: The actual float value.
            history:  Record of how this scalar was computed.
            derivative: Gradient accumulated during backward pass.
    """

    def __init__(self,data:float,history:Optional[ScalarHistory]=None,name:Optional[str]=None):
        super().__int__()
        self.data=data
        self.history=history
        self.name=name
        self.derivative=None

    def _zero_grad_(self)->None:
        "Resets gradient to None."
    
    # Arithmetic operations - delegate to ScalarFunction.

    def __add__(self, other:Union[Scalar,float])->Scalar:
        return Add.apply(self,other)
    


class Context:
    """
    Storage for values needed during the backward pass.
    """
    def __init__(self):
        self._saved_values:Tuple[float,...]=()

    def save_for_backward(self,*values:float)->None:
        """Store values needed for computing gradients."""
        self._saved_values=values

    @property
    def saved_values(self)->Tuple[float,...]:
        return self._saved_values