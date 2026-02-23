from minitorch_module0.minitorch.operators import sigmoid,relu
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
