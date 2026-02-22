import math
from typing import Callable,Iterable

def mul(x:float,y:float)->float:
    """ Multiply two numbers """
    return x*y

def id(x:float)->float:
    """ Identify function - returns input unchanged """
    identity_constant=1.0
    return identity_constant*x

def add(x:float,y:float)->float:
    """ Add two numbers """
    return x+y

def neg(x:float)->float:
    """Negate a number"""
    return -1.0*x

def lt(x:float,y:float)->float:
    """ less than comparison operation """
    return 1.0 if x<y else 0.0

def eq(x:float,y:float)->float:
    """ equality comparison operation """
    return 1.0 if x==y else 0.0

def max(x:float,y:float)->float:
    """ Return the max between x and y """
    return x if x>y else y

def is_close(x:float,y:float)->float:
    """Checks if two numbers are close enough """
    eps=1e-2
    return 1.0 if abs(x-y) < eps else 0.0


def sigmoid(x:float)->float:
    """
    Sigmoid activation function.
    Uses numerically stable implementation. 
    """
    if x>=0:
        return 1.0/(1.0+math.exp(-x))
    else:    
        exp_x=math.exp(x)
        return exp_x/(1.0+exp_x)
    
def relu(x:float)->float:
    """ReLu activation function : max(0,x) """
    return max(0,x)

# Mathemetical Function
def log(x:float)->float:
    """Natural Logarithm"""
    return math.log(x)

def exp(x:float)->float:
    """Exponential function: e^x"""
    return math.exp(x)

def inv(x:float)->float:
    return 1/x

# Gradient Helper function
def log_back(x:float,grad:float)->float:
    """Gradient of log(x) times incoming gradient.
        Derivative of log(x) is 1/x.
    """
    return grad/x

def inv_back(x:float,grad:float)->float:
    """
    Gradient of 1/x times incoming gradient.
    Derivative of 1/x is -1/x^2.
    """
    return -grad/(x*x)

def relu_back(x:float,grad:float)->float:
    """Graident of Relu times incoming gradient.
    Derivative is 1 if x>0, else 0
    """

    return grad if x>0 else 0.0


#Higher order functions
def map(fn:Callable[[float],float])->Callable[[Iterable[float]],Iterable[float]]:
    """
    Higher-order map function.
    Returns a function that applies fn to each element.
    """
    def mapped(ls: Iterable[float])->Iterable[float]:
        return [fn(x) for x in ls]
    return mapped


