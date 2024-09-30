import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
import pytensor.tensor.math as ptm
from pytensor.compile.io import In
from pytensor.configdefaults import config
from pytensor.graph.fg import FunctionGraph
from pytensor.tensor import elemwise as pt_elemwise
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.special import SoftmaxGrad, log_softmax, softmax
from pytensor.tensor.type import matrix, tensor, tensor3, vector
from tests.link.pytorch.test_basic import compare_pytorch_and_py


def test_pytorch_Dimshuffle():
    a_pt = matrix("a")

    x = a_pt.T
    x_fg = FunctionGraph([a_pt], [x])
    compare_pytorch_and_py(x_fg, [np.c_[[1.0, 2.0], [3.0, 4.0]].astype(config.floatX)])

    x = a_pt.dimshuffle([0, 1, "x"])
    x_fg = FunctionGraph([a_pt], [x])
    compare_pytorch_and_py(x_fg, [np.c_[[1.0, 2.0], [3.0, 4.0]].astype(config.floatX)])

    a_pt = tensor(dtype=config.floatX, shape=(None, 1))
    x = a_pt.dimshuffle((0,))
    x_fg = FunctionGraph([a_pt], [x])
    compare_pytorch_and_py(x_fg, [np.c_[[1.0, 2.0, 3.0, 4.0]].astype(config.floatX)])

    a_pt = tensor(dtype=config.floatX, shape=(None, 1))
    x = pt_elemwise.DimShuffle([False, True], (0,))(a_pt)
    x_fg = FunctionGraph([a_pt], [x])
    compare_pytorch_and_py(x_fg, [np.c_[[1.0, 2.0, 3.0, 4.0]].astype(config.floatX)])


def test_multiple_input_output():
    x = vector("x")
    y = vector("y")
    out = pt.mul(x, y)

    fg = FunctionGraph(outputs=[out], clone=False)
    compare_pytorch_and_py(fg, [[1.5], [2.5]])

    x = vector("x")
    y = vector("y")
    div = pt.int_div(x, y)
    pt_sum = pt.add(y, x)

    fg = FunctionGraph(outputs=[div, pt_sum], clone=False)
    compare_pytorch_and_py(fg, [[1.5], [2.5]])


def test_pytorch_elemwise():
    x = pt.vector("x")
    out = pt.log(1 - x)

    fg = FunctionGraph([x], [out])
    compare_pytorch_and_py(fg, [[0.9, 0.9]])


@pytest.mark.parametrize("fn", [ptm.sum, ptm.prod, ptm.max, ptm.min])
@pytest.mark.parametrize("axis", [None, 0, 1, (0, -1)])
def test_pytorch_careduce(fn, axis):
    a_pt = tensor3("a")
    test_value = np.array(
        [
            [
                [1, 1, 1, 1],
                [2, 2, 2, 2],
            ],
            [
                [3, 3, 3, 3],
                [
                    4,
                    4,
                    4,
                    4,
                ],
            ],
        ]
    ).astype(config.floatX)

    x = fn(a_pt, axis=axis)
    x_fg = FunctionGraph([a_pt], [x])

    compare_pytorch_and_py(x_fg, [test_value])


@pytest.mark.parametrize("fn", [ptm.any, ptm.all])
@pytest.mark.parametrize("axis", [None, 0, 1, (0, 1)])
def test_pytorch_any_all(fn, axis):
    a_pt = matrix("a")
    test_value = np.array([[True, False, True], [False, True, True]])

    x = fn(a_pt, axis=axis)
    x_fg = FunctionGraph([a_pt], [x])

    compare_pytorch_and_py(x_fg, [test_value])


@pytest.mark.parametrize("dtype", ["float64", "int64"])
@pytest.mark.parametrize("axis", [None, 0, 1])
def test_softmax(axis, dtype):
    x = matrix("x", dtype=dtype)
    out = softmax(x, axis=axis)
    fgraph = FunctionGraph([x], [out])
    test_input = np.arange(6, dtype=config.floatX).reshape(2, 3)

    if dtype == "int64":
        with pytest.raises(
            NotImplementedError,
            match="Pytorch Softmax is not currently implemented for non-float types.",
        ):
            compare_pytorch_and_py(fgraph, [test_input])
    else:
        compare_pytorch_and_py(fgraph, [test_input])


@pytest.mark.parametrize("dtype", ["float64", "int64"])
@pytest.mark.parametrize("axis", [None, 0, 1])
def test_logsoftmax(axis, dtype):
    x = matrix("x", dtype=dtype)
    out = log_softmax(x, axis=axis)
    fgraph = FunctionGraph([x], [out])
    test_input = np.arange(6, dtype=config.floatX).reshape(2, 3)

    if dtype == "int64":
        with pytest.raises(
            NotImplementedError,
            match="Pytorch LogSoftmax is not currently implemented for non-float types.",
        ):
            compare_pytorch_and_py(fgraph, [test_input])
    else:
        compare_pytorch_and_py(fgraph, [test_input])


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_softmax_grad(axis):
    dy = matrix("dy")
    dy_value = np.array([[1, 1, 1], [0, 0, 0]], dtype=config.floatX)
    sm = matrix("sm")
    sm_value = np.arange(6, dtype=config.floatX).reshape(2, 3)
    out = SoftmaxGrad(axis=axis)(dy, sm)
    fgraph = FunctionGraph([dy, sm], [out])
    compare_pytorch_and_py(fgraph, [dy_value, sm_value])


def test_fibonacci_loop():
    n_steps = pytensor.scalar.int32("n_steps")
    f0 = pytensor.scalar.float32("f0")
    f1 = pytensor.scalar.float32("f1")
    end = pytensor.scalar.float32("end")
    i = pytensor.scalar.float32("end")

    op = pytensor.scalar.ScalarLoop(
        init=[f0, f1, end, i],
        update=[
            pytensor.scalar.basic.identity(f1),
            f0 + f1,
            pytensor.scalar.basic.identity(end),
            i + 1,
        ],
        until=i >= end,
    )

    e = Elemwise(op)
    _, p, _, _, done = e(n_steps, f0, f1, end, i)

    fn = pytensor.function(
        [
            n_steps,
            end,
            In(f0, value=np.int32(0)),
            In(f1, value=np.int32(1)),
            In(i, value=np.int32(2)),
        ],
        [p, done],
        mode="PYTORCH",
    )

    res = ([[21.0], [5.0], [8.0]], [[0.0], [1.0], [1.0]])
    np.testing.assert_allclose(
        fn(np.array([[i] for i in [7, 5, 100]]), np.array([10.0, 5.0, 6.0])), res
    )
