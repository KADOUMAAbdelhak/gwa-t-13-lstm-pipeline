"""Per-VM random-forest predictor and residual stream generation."""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from dynapm.core.features import build_features


class VMPredictor:
    def __init__(
        self,
        target: str,
        n_estimators: int = 200,
        max_depth: int = 20,
        random_state: int = 42,
    ):
        self.target = target
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=1,
            max_features="sqrt",
        )
        self.feature_names_ = None

    def fit(self, df_bootstrap: pd.DataFrame) -> None:
        features = build_features(df_bootstrap, self.target)
        if len(features) == 0:
            raise ValueError("no training rows after feature construction")
        X = features.values
        y = df_bootstrap.loc[features.index, self.target].values
        self.model.fit(X, y)
        self.feature_names_ = list(features.columns)

    def predict_streaming(self, df_streaming: pd.DataFrame) -> pd.DataFrame:
        if self.feature_names_ is None:
            raise ValueError("model must be fit before predict_streaming")
        features = build_features(df_streaming, self.target)
        if len(features) == 0:
            return pd.DataFrame(columns=["y_true", "y_pred", "residual"], index=features.index)
        y_pred = self.model.predict(features.values)
        y_true = df_streaming.loc[features.index, self.target].values
        residual = y_true - y_pred
        return pd.DataFrame(
            {
                "y_true": y_true,
                "y_pred": y_pred,
                "residual": residual,
            },
            index=features.index,
        )
