import pytest
import math
from hypothesis import given
from hypothesis.strategies import floats
import torch

"""Unit tests"""
@pytest.mark.task0_1
def test_operators():
    from MyTorch.operators import (
        add,mul,neg,id
    )
    assert add(x=1.0,y=3.0)==4.0
    assert mul(x=1.5,y=4.0)==6.0
    assert neg(-10000.2)==10000.2
    assert id(-400.7)==-400.7


@pytest.mark.task0_1
def test_utility_fn():
    from MyTorch.operators import (
        lt,eq,max,is_close,log,exp,inv
    )
    assert lt(x=1.0,y=3.0)==1.0
    assert lt(x=5.0,y=3.0)==0.0
    assert eq(x=1.5,y=4.0)==0.0
    assert eq(x=1.5,y=1.5)==1.0
    assert max(x=3.0,y=1.0)==3.0
    assert max(x=1.0,y=3.0)==3.0
    assert is_close(x=1e-6,y=1e-6)==1.0
    assert is_close(x=1e-5,y=1e-4)==1.0
    assert is_close(x=1e-3,y=1e-6)==1.0
    assert is_close(x=4.0,y=3.0)==0.0
    assert log(x=3.45)==math.log(3.45)
    assert exp(x=234)==math.exp(234)
    assert inv(x=3.4)==(1/3.4)

@pytest.mark.task0_1
def test_activation_fn():
    from MyTorch.operators import (
        relu,sigmoid
    )
    from MyTorch.testing import assert_close

    assert relu(-1.3)==0
    assert relu(3.4)==3.4

    assert_close(sigmoid(4.5),torch.sigmoid(torch.tensor(4.5)).item())
    assert_close(sigmoid(-1000.0),torch.sigmoid(torch.tensor(-1000.0)).item())

@pytest.mark.task0_1
def test_grad_fn():
    from MyTorch.operators import (
        log_back,inv_back,relu_back
    )
    x=3.45
    grad=4.5
    assert log_back(x=x,grad=grad)==grad*(1/x)
    assert inv_back(x=x,grad=grad)==-grad*(1/(x*x))
    assert relu_back(x=x,grad=grad)==0 if x<0 else grad*1



"""Property based testing"""

small_floats=floats(min_value=-5,max_value=5)

@pytest.mark.task0_2
@given(small_floats)
def test_sigmoid(a):
    """Testing mathematical properties for sigmoid function"""
    from MyTorch.operators import sigmoid, is_close

    if math.isfinite(a):
        sig_a=sigmoid(a)
        # assert 0<sig_a<1
        assert 0<sig_a<1 # For neumerical Errors

        if is_close(a,0.0)==1.0:
            assert is_close(sig_a,0.5)==1.0

        sig_neg_a=sigmoid(-a)
        expected=1-sig_a
        assert is_close(sig_neg_a,expected)==1.0


@pytest.mark.task0_2
@given(small_floats,small_floats,small_floats)
def test_transitive(a,b,c):
    """Test transitive property: if a<b and b<c, then a<c."""
    from MyTorch.operators import lt

    if all(math.isfinite(v) for v in [a,b,c]):
        if lt(a,b)==True and lt(b,c)==True:
            assert lt(a,c)==True


@pytest.mark.task0_2
@given(small_floats,small_floats)
def test_symmetric(x,y):
    """Test that multiplication is commulative."""

    from MyTorch.operators import mul
    from MyTorch.testing import assert_close

    if math.isfinite(x) and math.isfinite(y):
        assert_close(mul(x,y),mul(y,x))


@pytest.mark.task0_2
@given(small_floats,small_floats,small_floats)
def test_distributive(x,y,z):
    """Test distributive propertry: z*(x+y) == z*x +z*y"""

    from MyTorch.operators import mul,add
    from MyTorch.testing import assert_close

    if all(math.isfinite(v) for v in [x,y,z]):
        left_side=mul(z,add(x,y))
        right_side=add(mul(z,x),mul(z,y))
        assert_close(left_side,right_side)

@pytest.mark.task0_2
@given(small_floats)
def test_other(a):
    """Test additive inverse property: a+(-a)=0."""
    from MyTorch.operators import add,neg
    from MyTorch.testing import assert_close

    if math.isfinite(a):
        result=add(a,neg(a))
        assert_close(result,0.0)
