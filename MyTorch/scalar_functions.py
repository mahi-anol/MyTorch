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
                last_fn=cls,
                ctx=ctx,
                inputs=scalars
            )
        else:
            history=None

        return Scalar(result_data,history=history)
    

    @staticmethod
    def forward(ctx:Context,*inputs:float)->float:
        """compute the function output. Override in subclass."""
        raise NotImplementedError()

    @staticmethod
    def backward(ctx:Context,d_output:float)->Tuple[float,...]:
        """
        Compute gradient with respect to inputs.

        Args:
            ctx: Context with saved values from forward.
            d_output: Gradients of loss with respect to output.

        Returns:
            Tuple of gradients, one per input.
        """
        raise NotImplementedError()
    
# Function Implementations.

class Add(ScalarFunction):
    """Addition: z=x+y"""

    @staticmethod
    def forward(ctx: Context,x:float,y:float)->float:
        return x+y

    @staticmethod
    def backward(ctx:Context,d_output:float)->Tuple[float,float]:
        return d_output,d_output
    

class Mul(ScalarFunction):
    """Multiplication: z=x*y"""

    @staticmethod
    def forward(ctx:Context,x:float,y:float):
        ctx.save_for_backward(x,y)
        return x*y
    
    @staticmethod
    def backward(ctx:Context,d_output:float)->Tuple[float,float]:
        x,y=ctx.saved_values
        return d_output*y,d_output*x
    

class Neg(ScalarFunction):
    """Negation: z=-x"""

    @staticmethod
    def forward(ctx:Context,x:float)->float:
        return -x
    
    @staticmethod
    def backward(ctx:Context,d_output:float)->Tuple[float]:
        return (-d_output,)
    

class Inv(ScalarFunction):
    """Inverse: z=1/x"""

    @staticmethod
    def forward(ctx: Context,x:float)->float:
        ctx.save_for_backward(x)
        return 1.0/x
    
    @staticmethod
    def backward(ctx:Context,d_output:float)->Tuple[float]:
        (x,)=ctx.saved_values
        return (-d_output/(x*x),)
    

class Log(ScalarFunction):
    """Natural log: z=log(x)"""
    @staticmethod
    def forward(ctx:Context,x:float)->float:
        ctx.save_for_backward(x)
        return math.log(x)

    @staticmethod
    def backward(ctx:Context,d_output:float)->Tuple[float]:
        (x,)=ctx.saved_values
        return (d_output/x,)
    

class Exp(ScalarFunction):
    """Exponential: z=e^x"""

    @staticmethod
    def forward(ctx:Context,x:float)->float:
        result=math.exp(x)
        ctx.save_for_backward(result) # saving output not input.
        return result
    
    def backward(ctx: Context,d_output:float)->Tuple[float]:
        (result,)=ctx.saved_values
        return (d_output,result,)
    

class Sigmoid(ScalarFunction):
    """sigmoid: z=1/(1+e^(-x))"""

    @staticmethod
    def forward(ctx:Context,x:float)->float:
        if x>0:
            result=1.0/(1.0+math.exp(-x))
        else:
            exp_x=math.exp(x)
            result=exp_x/(1.0+exp_x)
        ctx.save_for_backward(result)

    @staticmethod
    def backward(ctx:Context,d_output:float)->Tuple[float]:
        (result,)=ctx.saved_values
        return (d_output*result*(1.0-result),)
    

class ReLU(ScalarFunction):
    """ReLU: z=max(0,z)"""

    @staticmethod
    def forward(ctx:Context,x:float)->float:
        ctx.save_for_backward(x)
        return max(0.0,x)
    
    @staticmethod
    def backward(ctx:Context,d_output:float)->Tuple[float]:
        (x,)=ctx.saved_values
        #d(relu)/dx=1 if x>0 else 0
        return (d_output if x>0 else 0.0)
    
class Square(ScalarFunction):
    """Square: z = x^2"""

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        # What should you save?
        ctx.save_for_backward(x)
        return x*x

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float]:
        # Derivative of x^2 is 2x
        (x,)=ctx.saved_values
        return 2*x