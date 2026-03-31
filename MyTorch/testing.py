from MyTorch.operators import *
import math

""" Testing utilities for MiniTorch """
def assert_close(a:float,b:float)->None:
    """Assert two floats are close within tolerance."""
    assert abs(a-b)<1e-2, f"Values not close: {a} vs {b}"

print(f"sigmoid(0) = {sigmoid(0)}")
print(f"sigmoid(100) = {sigmoid(100)}")
print(f"sigmoid(-100) = {sigmoid(-100)}")
print(f"sigmoid(-5) = {sigmoid(-5)}")
print(f"sigmoid(5) = {sigmoid(5)}")


### The following code generates math range error.
# def naive_sigmoid(x: float) -> float:
#     """Naive sigmoid - NOT numerically stable."""
#     return 1.0 / (1.0 + math.exp(-x))

# # Test with extreme values
# print(naive_sigmoid(-1000))  # What happens?
result = sum(map(relu)([-1, 2, -3, 4]))
print("sum(map(relu)([-1, 2, -3, 4])) = ",result)

def dot(a,b):
    return reduce(add,0)(zipWith(mul)(a,b))

print("Dot Operation Output: ",dot([1,2,3],[4,5,6]))



def mean(ls: Iterable[float]) -> float:
    """Calculate the mean of a list."""
    return sum(ls)/len(ls)


from MyTorch.module import Module,Parameter

class Network(Module):
    def __init__(self):
        super().__init__()
        self.layer1=Module()
        self.layer1.weight=Parameter([[1,2],[3,4]])
        self.layer1.bias=Parameter([0.1,0.2])
        self.layer2=Module()
        self.layer2.weight=Parameter([[0.5]])


net=Network()
for name,_ in net.named_parameters():
    print(name)

print(net._parameters)


from MyTorch.datasets import simple

def simple_classifier(x: float, y: float) -> int:
    """Manual classifier for simple dataset."""
    return 1 if x >= 0.6 else 0 #0.5 gives 100% accuracy

# Test accuracy
X, y_true = simple(1000)
correct = sum(1 for (x, y), label in zip(X, y_true)
              if simple_classifier(x, y) == label)
accuracy = correct / len(X)
print(f"Accuracy: {accuracy:.2%}")



print("Central Difference.....")
from MyTorch.autodiff import central_difference

def square(x):
    return x * x

def mul(x, y):
    return x * y

print(central_difference(square, 3.0))           # Derivative of x² at x=3
print(central_difference(mul, 3.0, 4.0, arg=0))  # ∂(xy)/∂x at (3,4)
print(central_difference(mul, 3.0, 4.0, arg=1))  # ∂(xy)/∂y at (3,4)


print("Checking epsilon sensitivity....")

import math
from MyTorch.autodiff import central_difference

def exp_func(x):
    return math.exp(x)

# True derivative of e^x at x=1 is e^1 ≈ 2.718
true_derivative = math.e

for eps in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]:
    approx = central_difference(exp_func, 1.0, epsilon=eps)
    error = abs(approx - true_derivative)
    print(f"epsilon={eps:.0e}: approx={approx:.10f}, error={error:.2e}")


print("-----scalar testing------")


from MyTorch.scalar import Scalar

a = Scalar(2.0, name="a")
b = Scalar(3.0, name="b")

print(f"a.data = {a.data}")
print(f"a.is_leaf() = {a.is_leaf}")
print(f"a.derivative = {a.derivative}")

a.accumulate_derivative(1.0)
a.accumulate_derivative(2.0)
print(f"After accumulating: a.derivative = {a.derivative}")


## testing scalar
from MyTorch.scalar import Scalar

a = Scalar(2.0)
a.requires_grad_(True)
b = Scalar(3.0)
b.requires_grad_(True)

c = a * b  # c = 6.0

print(f"c.data = {c.data}")
print(f"c.is_leaf() = {c.is_leaf}")
print(f"c.history.last_fn = {c.history.last_fn}")


print("---Test scalar.backward()")

from MyTorch.scalar import Scalar

a=Scalar(2.0)
a.requires_grad_(True)
b=Scalar(3.0)
b.requires_grad_(True)

c=a*b
d=c+a
d.backward()

print(f"a.derivative = {a.derivative}")
print(f"b.derivative = {b.derivative}")


from MyTorch.scalar import Scalar
from MyTorch.autodiff import central_difference

def test_function(x_val, y_val):
    x = Scalar(x_val)
    x.requires_grad_(True)
    y = Scalar(y_val)
    y.requires_grad_(True)

    # Compute: z = (x * y) + x.log()
    z = x * y + x.log()
    z.backward()

    # Compare with numerical derivatives
    def f_for_x(x_val):
        return x_val * y_val + math.log(x_val)
    def f_for_y(y_val):
        return x_val * y_val + math.log(x_val)

    numerical_dx = central_difference(f_for_x, x_val)
    numerical_dy = central_difference(f_for_y, y_val)

    print(f"Autodiff: dx={x.derivative:.6f}, dy={y.derivative:.6f}")
    print(f"Numerical: dx={numerical_dx:.6f}, dy={numerical_dy:.6f}")

import math
test_function(2.0, 3.0)

print("----Broken Backprop----")


from MyTorch.autodiff import topological_sort
# DELIBERATELY BROKEN - processes in wrong order
def broken_backpropagate(variable, deriv=1.0):
    """Process nodes in forward order instead of reverse."""
    sorted_vars = topological_sort(variable)
    # BUG: Not reversing! Processing inputs before outputs

    variable.derivative = deriv

    for var in sorted_vars:  # Wrong order!
        if var.is_leaf or var.derivative is None:
            continue

        history = var.history
        if history is None or history.last_fn is None:
            continue

        backward_fn = history.last_fn.backward
        input_grads = backward_fn(history.ctx, var.derivative)

        for input_var, grad in zip(history.inputs, input_grads):
            if grad is not None:
                input_var.accumulate_derivative(grad)
a = Scalar(2.0); a.requires_grad_(True)
b = Scalar(3.0); b.requires_grad_(True)
c = a * b
d = c + a
broken_backpropagate(d, 1.0)

print(a.derivative)
print(b.derivative)


### Broadcast test
from MyTorch.tensor_data import shape_broadcast, broadcast_index
import numpy as np

# Experiment 1: Shape computation
print(shape_broadcast((3, 1), (1, 4)))  # Expected: (3, 4)
print(shape_broadcast((2, 3, 4), (4,))) # Expected: (2, 3, 4)

# Experiment 2: Try invalid broadcast
try:
    shape_broadcast((3, 2), (4, 2))
except Exception as e:
    print(f"Error: {e}")  # Expected: IndexingError

# Experiment 3: Index mapping
big_index = np.array([1, 2, 3])
out_index = np.array([0, 0])
broadcast_index(big_index, np.array([2, 3, 4]), np.array([1, 4]), out_index)
print(f"Mapped index: {out_index}")  # Expected: [0, 3]
