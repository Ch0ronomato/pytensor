import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.tensor.shape import specify_broadcastable


torch = pytest.importorskip("torch")


def test_blockwise_broadcast():
    _x = np.random.rand(5, 1, 2, 3)
    _y = np.random.rand(3, 3, 2)

    x = specify_broadcastable(pt.tensor4("x"), 1)
    y = pt.tensor3("y")

    f = pytensor.function([x, y], [x @ y], mode="PYTORCH")
    [res] = f(_x, _y)
    assert tuple(res.shape) == (5, 3, 2, 2)
    np.testing.assert_allclose(res, _x @ _y)
