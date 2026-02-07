from src.datasets import load_split, get_xy
from src.features import get_feature_spec_from_config_or_infer, build_preprocessor


def test_preprocessor_fit_transform_runs():
    df = load_split("train")
    X, y = get_xy(df)

    spec, target = get_feature_spec_from_config_or_infer(df)
    preprocessor = build_preprocessor(spec)

    Xt = preprocessor.fit_transform(X)
    assert Xt.shape[0] == X.shape[0]
    assert len(y) == X.shape[0]
