import json

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError
from scipy.special import expit

from mva.models.credit import CreditModelSpec
from mva.models.market import exceptions, portfolio_returns, rolling_var
from mva.package import ModelPackage, PackageError


def _model(**kw):
    base = dict(
        intercept=-2.0,
        coefficients={"income": -0.5, "util": 2.0},
        imputation={"income": 50000.0},
        caps={"util": (0.0, 1.0)},
        transforms={"income": "log"},
        expected_signs={"income": "-", "util": "+"},
    )
    base.update(kw)
    return CreditModelSpec(**base)


def test_preprocessing_and_scoring():
    m = _model()
    df = pd.DataFrame({"income": [np.nan, 100000.0], "util": [1.7, 0.4]})
    X = m.design_matrix(df)
    assert X.loc[0, "income"] == pytest.approx(np.log(50000))  # imputed then logged
    assert X.loc[0, "util"] == 1.0  # capped
    expected = expit(-2.0 - 0.5 * X["income"] + 2.0 * X["util"])
    assert np.allclose(m.predict_pd(df), expected)


def test_missing_input_without_imputation_rule_raises():
    m = _model(imputation={})
    with pytest.raises(ValueError, match="no imputation rule"):
        m.predict_pd(pd.DataFrame({"income": [np.nan], "util": [0.3]}))


def test_spec_rejects_rules_for_unknown_features():
    with pytest.raises(ValidationError):
        _model(expected_signs={"not_in_model": "+"})


def test_package_loads_model(credit_pkg):
    model = ModelPackage.load(credit_pkg).load_credit_model()
    assert "utilization_6m_avg" in model.features


def test_package_without_model_entry(credit_pkg):
    path = credit_pkg / "model_package.yaml"
    manifest = yaml.safe_load(path.read_text())
    del manifest["model"]
    path.write_text(yaml.safe_dump(manifest))
    with pytest.raises(PackageError, match="no 'model' entry"):
        ModelPackage.load(credit_pkg).load_credit_model()


def test_model_feature_must_be_declared(credit_pkg):
    path = credit_pkg / "model.json"
    model = json.loads(path.read_text())
    model["coefficients"]["shoe_size"] = 0.1
    path.write_text(json.dumps(model))
    with pytest.raises(PackageError, match="shoe_size"):
        ModelPackage.load(credit_pkg).load_credit_model()


def test_portfolio_returns_discloses_treatment(market_pkg):
    pkg = ModelPackage.load(market_pkg)
    returns, notes = portfolio_returns(pkg.load_data()["prices"], pkg.spec)
    assert returns.index.is_unique and returns.index.is_monotonic_increasing
    assert not returns.isna().any()
    text = " ".join(notes)
    assert "duplicated date" in text and "non-positive" in text and "forward-filled" in text


def test_rolling_var_has_no_look_ahead():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0, 0.01, 400), index=pd.bdate_range("2024-01-01", periods=400))
    var = rolling_var(r, 250, 0.99, "historical")
    r2 = r.copy()
    r2.iloc[-1] = -0.5  # a crash on the last day must not change that day's forecast
    var2 = rolling_var(r2, 250, 0.99, "historical")
    assert var.iloc[-1] == var2.iloc[-1]
    hits = exceptions(r2, var2)
    assert hits["exception"].iloc[-1] == 1
    assert len(hits) == 400 - 250
