import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor.compile.function import function
from pytensor.compile.sharedvalue import shared
from tests.link.pytorch.test_basic import pytorch_mode


torch = pytest.importorskip("torch")


def test_random_updates():
    original = np.random.default_rng(seed=123)
    rng = shared(original, name="rng", borrow=False)
    rv = pt.random.bernoulli(0.5, name="y", rng=rng)
    next_rng, x = rv.owner.outputs
    x.dprint()
    f = function([], [x], updates={rng: next_rng}, mode="PYTORCH")
    assert any(f() for _ in range(5))

    assert all(
        a == b if not isinstance(a, np.ndarray) else np.array_equal(a, b)
        for a, b in zip(
            rng.get_value(),
            original.bit_generator.state,
            strict=True,
        )
    )


@pytest.mark.parametrize(
    "size,p",
    [
        ((1000,), 0.5),
        (None, 0.5),
        ((1000, 4), 0.5),
        ((10, 2), np.array([0.5, 0.3])),
        ((1000, 10, 2), np.array([0.5, 0.3])),
    ],
)
def test_random_bernoulli(size, p):
    rng = shared(np.random.default_rng(123))

    g = pt.random.bernoulli(p, size=size, rng=rng)
    g_fn = function([], g, mode=pytorch_mode)
    samples = g_fn()
    np.testing.assert_allclose(samples.mean(axis=0), 0.5, 1)


@pytest.mark.parametrize(
    "size,n,p",
    [
        (None, 10, 0.5),
        ((1000,), 10, 0.5),
        ((1000, 4), 10, 0.5),
        ((1000, 2), np.array([10, 40]), np.array([0.5, 0.3])),
    ],
)
def test_binomial(n, p, size):
    rng = shared(np.random.default_rng(123))
    g = pt.random.binomial(n, p, size=size, rng=rng)
    g_fn = function([], g, mode=pytorch_mode)
    samples = g_fn()
    if size:
        np.testing.assert_allclose(samples.mean(axis=0), n * p, rtol=0.1)
        np.testing.assert_allclose(
            samples.std(axis=0), np.sqrt(n * p * (1 - p)), rtol=0.2
        )
    else:
        ...
        # TODO: define test
