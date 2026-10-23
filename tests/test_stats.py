import numpy as np
import pytest
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from mva import stats

rng = np.random.default_rng(42)


def test_fit_logit_matches_sklearn_unpenalised():
    n = 5000
    X = np.column_stack([rng.normal(size=n), rng.normal(50, 10, size=n)])
    y = rng.binomial(1, expit(-1 + 0.8 * X[:, 0] + 0.03 * (X[:, 1] - 50)))
    res = stats.fit_logit(X, y)
    sk = LogisticRegression(C=1e10, max_iter=1000).fit(X, y)  # effectively unpenalised
    assert res.converged
    assert np.allclose(res.coef, sk.coef_[0], rtol=1e-3, atol=1e-5)
    assert res.intercept == pytest.approx(sk.intercept_[0], rel=1e-3)
    assert np.all(res.se > 0)


def test_lr_test():
    assert stats.lr_test(-100.0, -100.0, 1) == pytest.approx(1.0)
    assert stats.lr_test(-100.0, -90.0, 1) < 1e-4


def test_vif():
    import pandas as pd

    a = rng.normal(size=2000)
    df = pd.DataFrame({"a": a, "b": rng.normal(size=2000), "c": a + rng.normal(0, 0.05, 2000)})
    v = stats.vif(df)
    assert v["b"] == pytest.approx(1.0, abs=0.05)
    assert v["a"] > 100 and v["c"] > 100


def test_kupiec_known_value():
    # 10 exceptions in 250 days at 99%: LR = 12.96 (Jorion, Value at Risk, ch. 6)
    lr, p = stats.kupiec_pof(10, 250, 0.01)
    assert lr == pytest.approx(12.96, abs=0.01)
    assert p < 0.001
    lr, p = stats.kupiec_pof(3, 250, 0.01)
    assert p > 0.5  # 3 exceptions vs 2.5 expected is unremarkable
    lr, p = stats.kupiec_pof(0, 250, 0.01)
    assert lr == pytest.approx(-2 * 250 * np.log(0.99))  # 0 exceptions: LR = 5.03, p = 0.025


@pytest.mark.parametrize("x, zone", [(0, "green"), (4, "green"), (5, "yellow"), (9, "yellow"), (10, "red")])
def test_traffic_light_reproduces_basel_zones(x, zone):
    assert stats.traffic_light(x, 250, 0.01)[0] == zone


def test_christoffersen_detects_clustering():
    hits = np.zeros(500, dtype=int)
    hits[[100, 101, 102, 103, 300, 301, 302]] = 1
    _, p, counts = stats.christoffersen_independence(hits)
    assert counts["n11"] == 5 and p < 0.001
    assert stats.christoffersen_independence(np.zeros(100))[1] == 1.0


def test_ljung_box():
    iid = rng.normal(size=1000)
    assert stats.ljung_box(iid, 10)[1] > 0.01
    ar = np.zeros(1000)
    e = rng.normal(size=1000)
    for t in range(1, 1000):
        ar[t] = 0.5 * ar[t - 1] + e[t]
    assert stats.ljung_box(ar, 10)[1] < 1e-6


def test_hosmer_lemeshow_and_calibration_level():
    p = rng.uniform(0.01, 0.3, 20000)
    y = rng.binomial(1, p)
    _, hl_p, table = stats.hosmer_lemeshow(p, y)
    assert hl_p > 0.01 and len(table) == 10
    assert abs(stats.calibration_in_the_large(p, y)[0]) < 2.58

    y_bad = rng.binomial(1, np.clip(p * 1.5, 0, 1))
    assert stats.hosmer_lemeshow(p, y_bad)[1] < 1e-6
    z, pval = stats.calibration_in_the_large(p, y_bad)
    assert z > 0 and pval < 1e-6  # more defaults than predicted


def test_ks_statistic():
    y = np.array([0] * 500 + [1] * 500)
    scores = np.concatenate([rng.uniform(0, 0.5, 500), rng.uniform(0.5, 1, 500)])
    assert stats.ks_statistic(scores, y) == pytest.approx(1.0)
