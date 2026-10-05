"""Any registered probability classifier can consume either feature dimension."""

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def make_model(name, params, seed=42, threads=4):
    params = dict(params)
    if name == "logistic":
        classifier = LogisticRegression(random_state=seed, **params)
        scale = StandardScaler()
    elif name == "hist_boosting":
        if params.get("early_stopping", False) is not False:
            raise ValueError("Automatic random validation is not allowed; use early_stopping=False")
        params["early_stopping"] = False
        classifier = HistGradientBoostingClassifier(random_state=seed, **params)
        scale = "passthrough"
    elif name == "random_forest":
        if params.get("oob_score", False):
            raise ValueError("Read-level OOB evaluation is not independent")
        classifier = RandomForestClassifier(random_state=seed, n_jobs=threads, **params)
        scale = "passthrough"
    elif name == "bagging":
        classifier = BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=12),
            random_state=seed,
            n_jobs=threads,
            **params,
        )
        scale = "passthrough"
    elif name == "dummy":
        classifier = DummyClassifier(strategy="prior")
        scale = "passthrough"
    else:
        raise ValueError(f"Unknown classifier: {name}")
    return Pipeline([("scale", scale), ("classifier", classifier)])


def positive_probability(model, features):
    classes = list(model.classes_)
    if classes != [0, 1]:
        raise ValueError(f"Expected binary classes [0, 1], got {classes}")
    return model.predict_proba(features)[:, classes.index(1)]
