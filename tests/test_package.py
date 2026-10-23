import pytest
import yaml

from mva.package import CreditPDSpec, MarketVaRSpec, ModelPackage, PackageError


def test_loads_credit_package(credit_pkg):
    pkg = ModelPackage.load(credit_pkg)
    assert pkg.model_type == "credit_pd"
    assert isinstance(pkg.spec, CreditPDSpec)
    data = pkg.load_data()
    assert set(data) == {"train", "test"}
    assert pkg.spec.target in data["train"].columns


def test_loads_market_package(market_pkg):
    pkg = ModelPackage.load(market_pkg)
    assert isinstance(pkg.spec, MarketVaRSpec)
    prices = pkg.load_data()["prices"]
    assert str(prices["date"].dtype).startswith("datetime64")
    assert pkg.spec.var.confidence == 0.99


def test_missing_manifest_raises(tmp_path):
    with pytest.raises(PackageError, match="No model_package.yaml"):
        ModelPackage.load(tmp_path)


def test_weights_must_match_assets(market_pkg):
    path = market_pkg / "model_package.yaml"
    manifest = yaml.safe_load(path.read_text())
    del manifest["weights"]["FX_EURUSD"]
    path.write_text(yaml.safe_dump(manifest))
    with pytest.raises(PackageError, match="weights must cover"):
        ModelPackage.load(market_pkg)


def test_declared_feature_missing_from_data(credit_pkg):
    path = credit_pkg / "model_package.yaml"
    manifest = yaml.safe_load(path.read_text())
    manifest["features"].append("not_a_column")
    path.write_text(yaml.safe_dump(manifest))
    pkg = ModelPackage.load(credit_pkg)
    with pytest.raises(PackageError, match="not_a_column"):
        pkg.load_data()


def test_unknown_model_type_rejected(credit_pkg):
    path = credit_pkg / "model_package.yaml"
    manifest = yaml.safe_load(path.read_text())
    manifest["model_type"] = "pricing"
    path.write_text(yaml.safe_dump(manifest))
    with pytest.raises(PackageError):
        ModelPackage.load(credit_pkg)
