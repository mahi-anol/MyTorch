"""
Scalar functions with forward and backward methods.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Tuple,Union


if TYPE_CHECKING:
    from MyTorch.scalar import Scalar,Context


import math

class ScalarFunction:
    """
    Base class for differentiable scalar operations.

    Subclasses Contains:
        forward: Compute result from inputs.
        backward: Compute gradient with respect to inputs.
    """

    @classmethod
    def apply(cls,*raw_vals:Union[Scalar,float])->Scalar:
        """
        Apply the function and record history for autodiff.
        """
        from MyTorch.scalar import Scalar,ScalarHistory,Context

        scalars=[]
        for val in raw_vals:
            if isinstance(val,Scalar):
                scalars.append(val)
            else:
                # Wrap float as a scalar (no gradient tracking)
                scalars.append(Scalar(float(val)))

        # Extract raw float values for forward
        raw_data=[s.data for s in scalars]

        # Create context to save values for backward
        ctx=Context()

        # Call forward - subclass implements this 
        result_data=cls.forward(ctx,*raw_data)

        # Check if any input requires gradient.
        requires_grad=any(s.history is not None for s in scalars)

        # Create result Scalar
        if requires_grad:
            history=ScalarHistory(
                last_fn=s.history
                ctx=Context
            )
        else:
            history=None

        return Scalar(result_data,history=history)