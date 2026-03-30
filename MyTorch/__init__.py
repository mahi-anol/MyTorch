from .tensor_data import TensorData,IndexingError,shape_broadcast
from .operators import prod
from .tensor import Tensor
from .tensor_functions import tensor
from .testing import MathTestVariable, grad_check
from .tensor_ops import SimpleBackend
from .module import Module,Parameter