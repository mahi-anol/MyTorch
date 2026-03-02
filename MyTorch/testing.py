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
print(f"a.is_leaf() = {a.is_leaf()}")
print(f"a.derivative = {a.derivative}")

a.accumulate_derivative(1.0)
a.accumulate_derivative(2.0)
print(f"After accumulating: a.derivative = {a.derivative}")


## testing scalar
from MyTorch.scalar import Scalar

a = Scalar(2.0)
a.requires_grad(True)
b = Scalar(3.0)
b.requires_grad(True)

c = a * b  # c = 6.0

print(f"c.data = {c.data}")
print(f"c.is_leaf() = {c.is_leaf()}")
print(f"c.history.last_fn = {c.history.last_fn}")
