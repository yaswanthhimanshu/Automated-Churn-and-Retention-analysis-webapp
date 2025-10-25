"""
core.py 

Key points:
- Robust reader for CSV/XLSX bytes (chardet + fallbacks).
- EDA: quick and full (numeric summary, correlation, top categories).
- Preprocessing: stable categorical encoding (mapping), median imputation, standard scaling.
- Model factory: many sklearn models, optional XGBoost / CatBoost if installed.
- train_model supports "split" and "full" modes, safe test_size handling and stratify checks.
- Predict, row explain (SHAP best-effort), and simulate ROI utilities.
- No disk writes. All objects remain in memory (models have attached preprocessing artifacts).
"""

from typing import Tuple, Dict, Any, Optional
import io
import warnings
import traceback

import chardet
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Optional shap import (best-effort)
try:
    import shap  # type: ignore
    SHAP_AVAILABLE = True
    warnings.filterwarnings("ignore", category=UserWarning)
except Exception:
    shap = None
    SHAP_AVAILABLE = False

# Optional heavy libraries (best-effort)
try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:
    XGBClassifier = None

try:
    from catboost import CatBoostClassifier  # type: ignore
except Exception:
    CatBoostClassifier = None

# ---------------------------
# Safe file reading (CSV/XLSX)
# ---------------------------
def safe_read_csv_bytes(raw_bytes: bytes) -> Optional[pd.DataFrame]:
    """
    Robust reader for CSV / XLSX bytes:
      - detects XLSX by PK header and tries pd.read_excel
      - uses chardet to guess encoding, falls back to common encodings
      - returns DataFrame or None on failure
    Attaches _detected_encoding attribute to returned DataFrame.
    """
    try:
        if raw_bytes is None:
            return None

        # Quick XLSX detection (PK..)
        if isinstance(raw_bytes, (bytes, bytearray)) and raw_bytes[:4] == b'PK\x03\x04':
            try:
                df = pd.read_excel(io.BytesIO(raw_bytes))
                df._detected_encoding = "xlsx"
                return df
            except Exception:
                # fall back to CSV parsing
                pass

        guess = chardet.detect(raw_bytes or b"")
        detected = guess.get("encoding") or ""
        candidates = [detected] if detected else []
        candidates += ["utf-8-sig", "utf-8", "cp1252", "latin1"]

        last_exc = None
        for enc in candidates:
            if not enc:
                continue
            try:
                text = raw_bytes.decode(enc)
                df = pd.read_csv(io.StringIO(text), low_memory=False)
                df._detected_encoding = enc
                return df
            except Exception as e:
                last_exc = e
                continue

        # final fallback decode latin1 with replacement
        try:
            text = raw_bytes.decode("latin1", errors="replace")
            df = pd.read_csv(io.StringIO(text), low_memory=False)
            df._detected_encoding = "latin1_replace"
            return df
        except Exception:
            return None
    except Exception:
        return None

# ---------------------------
# Preview generator
# ---------------------------
def preview_df_html(df: pd.DataFrame, max_rows: int = 1000, max_cols: int = 1000) -> str:
    """
    Return safe HTML preview capped to max_rows x max_cols.
    Truncates very long strings for readability.
    """
    try:
        r = min(max_rows, len(df))
        c = min(max_cols, df.shape[1])
        preview = df.head(r).iloc[:, :c].copy()
        for col in preview.select_dtypes(include=["object"]).columns:
            preview[col] = preview[col].apply(
                lambda x: (str(x)[:200] + "...") if isinstance(x, str) and len(str(x)) > 200 else x
            )
        return preview.to_html(classes="table table-sm table-striped", index=False, escape=True)
    except Exception as e:
        return f"<div class='error'>Preview generation failed: {e}</div>"

# ---------------------------
# EDA
# ---------------------------
def quick_eda(df: pd.DataFrame) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    info["n_rows"], info["n_cols"] = int(df.shape[0]), int(df.shape[1])
    cols = []
    for c in df.columns:
        dtype = str(df[c].dtype)
        missing = int(df[c].isna().sum())
        cols.append({"name": c, "dtype": dtype, "missing": missing})
    info["columns"] = cols
    info["missing_total"] = int(df.isna().sum().sum())
    info["numeric_cols"] = [c for c in df.select_dtypes(include=[np.number]).columns.tolist()]
    info["categorical_cols"] = [c for c in df.columns if c not in info["numeric_cols"]]
    return info

def generate_full_report(df: pd.DataFrame, sample_limit: int = 50000) -> Dict[str, Any]:
    """
    Return a heavier EDA report:
      - numeric_summary (describe)
      - correlation (numeric)
      - top_categories for categorical columns
    Uses sampling when df larger than sample_limit.
    """
    n = len(df)
    sample = df.sample(n=sample_limit, random_state=42) if n > sample_limit else df.copy()
    report = quick_eda(sample)
    numeric_cols = report["numeric_cols"]
    if numeric_cols:
        try:
            report["correlation"] = sample[numeric_cols].corr().round(3).to_dict()
        except Exception:
            report["correlation"] = {}
        try:
            report["numeric_summary"] = sample[numeric_cols].describe().round(3).to_dict()
        except Exception:
            report["numeric_summary"] = {}

    cat_cols = report["categorical_cols"]
    if cat_cols:
        top_vals = {}
        for c in cat_cols:
            try:
                top_vals[c] = sample[c].value_counts(normalize=True).head(10).round(3).to_dict()
            except Exception:
                top_vals[c] = {}
        report["top_categories"] = top_vals

    return report

# ---------------------------
# Categorical encoder utilities
# ---------------------------
def _fit_categorical_encoders(X: pd.DataFrame) -> Dict[str, Dict[Any, int]]:
    """
    For each non-numeric column create a stable mapping value -> int.
    Unknown categories at transform time map to -1.
    """
    encoders: Dict[str, Dict[Any, int]] = {}
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            vals = X[col].astype(str).fillna("__MISSING__")
            uniques = pd.Series(vals.unique()).tolist()
            mapping = {u: i for i, u in enumerate(uniques)}
            encoders[col] = mapping
    return encoders

def _transform_with_encoders(X: pd.DataFrame, encoders: Dict[str, Dict[Any, int]]) -> pd.DataFrame:
    Xt = X.copy()
    for col, mapping in (encoders or {}).items():
        if col in Xt.columns:
            vals = Xt[col].astype(str).fillna("__MISSING__")
            Xt[col] = vals.map(lambda v: mapping.get(v, -1)).astype(float)
    # For numeric columns not in encoders, keep original values
    # Ensure order of columns remains the same as input X
    return Xt

# ---------------------------
# Preprocessing for model
# ---------------------------
def preprocess_for_model(df: pd.DataFrame, target_col: str, fit: bool = True,
                         encoders: Optional[Dict[str, Dict[Any, int]]] = None,
                         imputer: Optional[SimpleImputer] = None,
                         scaler: Optional[StandardScaler] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Dict[Any, int]], SimpleImputer, StandardScaler]:
    """
    Returns (X_scaled, y_array, encoders, imputer, scaler).
    X_scaled is numpy array of shape (n_samples, n_features) where features correspond to df.drop(target_col).columns
    """
    if target_col not in df.columns:
        raise ValueError(f"target_col {target_col} not in dataframe")

    dfc = df.copy()
    y = dfc[target_col].values
    X = dfc.drop(columns=[target_col])

    if fit or encoders is None:
        encoders = _fit_categorical_encoders(X)
    X_enc = _transform_with_encoders(X, encoders)

    if fit or imputer is None:
        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(X_enc)
    else:
        X_imputed = imputer.transform(X_enc)

    if fit or scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
    else:
        X_scaled = scaler.transform(X_imputed)

    return X_scaled, np.asarray(y), encoders, imputer, scaler

# ---------------------------
# Model factory 
# ---------------------------
def get_model(model_type: str):
    mt = (model_type or "randomforest").lower().replace(" ", "")
    models = {
        "logistic": LogisticRegression(max_iter=1000, solver="lbfgs"),
        "catboost": CatBoostClassifier(iterations=300, learning_rate=0.1, depth=6, verbose=False, allow_writing_files=False),
        "randomforest": RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
        "decisiontree": DecisionTreeClassifier(random_state=42),
        "gradientboost": GradientBoostingClassifier(random_state=42),
        "adaboost": AdaBoostClassifier(random_state=42),
        "naivebayes": GaussianNB(),
        "xgboost": XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42, use_label_encoder=False, eval_metric="logloss"),
    }
    return models.get(mt, models["randomforest"])


# ---------------------------
# Internal metrics helper
# ---------------------------
def _compute_metrics(y_true, y_pred, y_proba=None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        out["accuracy"] = float(accuracy_score(y_true, y_pred))
        out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    except Exception:
        from sklearn.metrics import precision_score as psc, recall_score as rsc, f1_score as fsc
        out["precision"] = float(psc(y_true, y_pred, average="weighted", zero_division=0))
        out["recall"] = float(rsc(y_true, y_pred, average="weighted", zero_division=0))
        out["f1"] = float(fsc(y_true, y_pred, average="weighted", zero_division=0))
    out["roc_auc"] = None
    if y_proba is not None:
        try:
            if hasattr(y_proba, "ndim") and y_proba.ndim == 2 and y_proba.shape[1] == 2:
                out["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
        except Exception:
            out["roc_auc"] = None
    return out

# ---------------------------
# Train model (robust)
# ---------------------------
def train_model(df: pd.DataFrame,
                target_col: str,
                model_type: str = "logistic",
                sample_ratio: Optional[float] = 0.2,
                mode: str = "split",
                compute_cv: bool = False,
                cv_folds: int = 3) -> Tuple[str, Dict[str, Any], Any]:
    """
    Train and return (model_id, meta_dict, model_obj).
    meta_dict contains status and metrics or an error key on failure.
    mode: "split" (train/test split) or "full" (train on all rows).
    sample_ratio: when mode=='split' it's the test_size; validated to (0,1).
    """
    try:
        if target_col not in df.columns:
            raise ValueError("target_col not found in dataframe")

        df_clean = df.dropna(subset=[target_col]).copy()
        if df_clean.shape[0] == 0:
            raise ValueError("no rows with non-null target to train on")

        # Preprocess using all columns except target (fit)
        X_all_scaled, y_all, encoders, imputer, scaler = preprocess_for_model(df_clean, target_col, fit=True)

        # Ensure sample_ratio is valid only when in split mode
        if mode == "split":
            try:
                sample_ratio = float(sample_ratio)
            except Exception:
                sample_ratio = 0.2
            if not (0.0 < sample_ratio < 1.0):
                sample_ratio = 0.2
        else:
            # ignore sample_ratio when mode == full
            sample_ratio = None

        # select estimator
        estimator = get_model(model_type)

        if mode == "full":
            # train on all rows
            estimator.fit(X_all_scaled, y_all)
            try:
                y_pred = estimator.predict(X_all_scaled)
            except Exception:
                y_pred = np.zeros_like(y_all)
            try:
                y_proba = estimator.predict_proba(X_all_scaled)
            except Exception:
                y_proba = None

            metrics = _compute_metrics(y_all, y_pred, y_proba)

            if compute_cv:
                try:
                    cv_scores = cross_val_score(estimator, X_all_scaled, y_all, cv=min(cv_folds, 5), scoring="accuracy", n_jobs=-1)
                    metrics["cv_mean"] = float(np.mean(cv_scores))
                    metrics["cv_std"] = float(np.std(cv_scores))
                except Exception:
                    pass

            model_obj = estimator
            model_obj.encoders = encoders
            model_obj.imputer = imputer
            model_obj.scaler = scaler
            model_obj.target_col = target_col
            model_id = f"{model_type}_{int(np.random.randint(1000, 9999))}"
            return model_id, {"status": "success", "trained_on": "full", "metrics": metrics, "n_rows": int(len(y_all))}, model_obj

        # ---------------------------
        # split mode: safe stratify
        # ---------------------------
        stratify = None
        try:
            if np.unique(y_all).size > 1:
                vals, counts = np.unique(y_all, return_counts=True)
                if np.min(counts) >= 2:
                    stratify = y_all
        except Exception:
            stratify = None

        X_train, X_test, y_train, y_test = train_test_split(
            X_all_scaled, y_all, test_size=float(sample_ratio or 0.2), random_state=42, stratify=stratify
        )

        estimator.fit(X_train, y_train)
        try:
            y_pred = estimator.predict(X_test)
        except Exception:
            y_pred = np.zeros_like(y_test)
        try:
            y_proba = estimator.predict_proba(X_test)
        except Exception:
            y_proba = None

        metrics = _compute_metrics(y_test, y_pred, y_proba)

        model_obj = estimator
        model_obj.encoders = encoders
        model_obj.imputer = imputer
        model_obj.scaler = scaler
        model_obj.target_col = target_col

        model_id = f"{model_type}_{int(np.random.randint(1000, 9999))}"
        meta = {
            "status": "success",
            "trained_on": "split",
            "test_size": float(sample_ratio or 0.2),
            "metrics": metrics,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "n_rows": int(len(y_train))
        }

        return model_id, meta, model_obj

    except Exception as exc:
        traceback.print_exc()
        return "", {"status": "error", "error": str(exc)}, None

# ---------------------------
# Prediction
# ---------------------------
def predict_df(model: Any, df: pd.DataFrame) -> pd.DataFrame:
    """
    Use trained model to produce predictions for df.
    Returns copy of df with:
      - 'predicted_churn' (0/1 integer)
      - 'churn_probability' (float in [0,1])
    Defensive: accepts a df that may still contain the target column (string labels like "Yes"/"No")
    and never attempts unsafe int() conversions.
    """
    if not hasattr(model, "encoders") or not hasattr(model, "imputer") or not hasattr(model, "scaler"):
        raise ValueError("Model missing preprocessing artifacts. Train model using train_model first.")

    df_copy = df.copy()

    # If the uploaded scoring file contains the original target column, drop it before transforming
    if hasattr(model, "target_col") and model.target_col in df_copy.columns:
        df_proc = df_copy.drop(columns=[model.target_col])
    else:
        df_proc = df_copy

    # Transform with the encoders (unknown -> -1)
    X_enc = _transform_with_encoders(df_proc, model.encoders or {})
    # Impute / scale (use model's artifacts)
    X_imputed = model.imputer.transform(X_enc)
    X_scaled = model.scaler.transform(X_imputed)

    # Predict (safe)
    try:
        preds = model.predict(X_scaled)
    except Exception:
        # fallback: all zeros
        preds = np.zeros((X_scaled.shape[0],), dtype=int)

    # Probability (best-effort)
    probs = None
    try:
        proba = model.predict_proba(X_scaled)
        # picks the probability for the positive class if available
        if hasattr(proba, "ndim") and proba.ndim == 2 and proba.shape[1] >= 2:
            probs = proba[:, 1].astype(float)
        else:
            # single-column probability or unsupported shape -> coerce to 0/1-like floats
            probs = np.asarray(proba).reshape(-1).astype(float)
    except Exception:
        # If predict_proba not available, derive from preds (0/1)
        try:
            probs = preds.astype(float)
        except Exception:
            probs = np.zeros((len(preds),), dtype=float)

    # Build output safely
    out = df_copy.copy()
    # Ensures numeric types and safe casting
    out["churn_probability"] = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    # Some estimators return non-zero/non-one preds; coerce to 0/1 ints
    try:
        out["predicted_churn"] = np.where(np.asarray(preds, dtype=float) >= 0.5, 1, 0).astype(int)
    except Exception:
        # fallback: if preds are not numeric, try to map common strings -> 0/1
        mapped = []
        for v in preds:
            try:
                mapped.append(int(v))
            except Exception:
                sv = str(v).strip().lower()
                mapped.append(1 if sv in ("1", "yes", "true", "y", "churn") else 0)
        out["predicted_churn"] = np.asarray(mapped, dtype=int)

    return out

# ---------------------------
# SHAP summary (best-effort)
# ---------------------------
def compute_shap_summary(model: Any, df: pd.DataFrame, sample_limit: int = 5000) -> Optional[Dict[str, Any]]:
    """
    Return top features by mean absolute SHAP value (best-effort).
    Returns None if shap not available or explainer fails.
    """
    if not SHAP_AVAILABLE:
        return None
    if not hasattr(model, "encoders"):
        return None
    try:
        n = len(df)
        sample = df.sample(n=min(sample_limit, n), random_state=42) if n > sample_limit else df.copy()
        if model.target_col in sample.columns:
            sample_proc = sample.drop(columns=[model.target_col])
        else:
            sample_proc = sample

        X_enc = _transform_with_encoders(sample_proc, model.encoders)
        X_imputed = model.imputer.transform(X_enc)
        X_scaled = model.scaler.transform(X_imputed)

        try:
            # prefer tree explainer for tree models
            if hasattr(model, "feature_importances_") or (XGBClassifier is not None and isinstance(model, XGBClassifier)):
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X_scaled)
                if isinstance(shap_vals, list) and len(shap_vals) > 1:
                    vals = np.abs(shap_vals[1]).mean(axis=0)
                else:
                    vals = np.abs(shap_vals).mean(axis=0)
            else:
                explainer = shap.Explainer(model, X_scaled)
                sv = explainer(X_scaled)
                vals = np.abs(sv.values).mean(axis=0)
        except Exception:
            try:
                explainer = shap.Explainer(model.predict, X_scaled)
                sv = explainer(X_scaled)
                vals = np.abs(sv.values).mean(axis=0)
            except Exception:
                return None

        feature_names = list(sample_proc.columns)
        shap_summary = [{"name": fn, "mean_abs_shap": float(v)} for fn, v in zip(feature_names, vals)]
        shap_summary = sorted(shap_summary, key=lambda x: x["mean_abs_shap"], reverse=True)
        return {"top_features": shap_summary[:min(len(shap_summary), 50)]}
    except Exception:
        return None

# ---------------------------
# Row-level explanation
# ---------------------------
def explain_row(model: Any, row_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Return SHAP-based explanation for a single-row DataFrame (or fallback).
    Returns dict with feature_contributions, predicted_churn and churn_probability.
    """
    if not hasattr(model, "encoders") or not SHAP_AVAILABLE:
        preds_df = predict_df(model, row_df)
        row = preds_df.iloc[0]
        return {
            "feature_contributions": None,
            "predicted_churn": int(row["predicted_churn"]),
            "churn_probability": float(row["churn_probability"])
        }
    try:
        df_proc = row_df.copy()
        if model.target_col in df_proc.columns:
            df_proc = df_proc.drop(columns=[model.target_col])
        X_enc = _transform_with_encoders(df_proc, model.encoders)
        X_imputed = model.imputer.transform(X_enc)
        X_scaled = model.scaler.transform(X_imputed)

        if hasattr(model, "feature_importances_") or isinstance(model, RandomForestClassifier):
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_scaled)
            if isinstance(shap_vals, list) and len(shap_vals) > 1:
                vals = shap_vals[1][0]
            else:
                vals = shap_vals[0]
        else:
            explainer = shap.Explainer(model, X_scaled)
            sv = explainer(X_scaled)
            vals = sv.values[0]

        feature_names = list(df_proc.columns)
        contributions = {fn: float(v) for fn, v in zip(feature_names, vals)}
        preds = model.predict(X_scaled)[0]
        try:
            prob = float(model.predict_proba(X_scaled)[0][1])
        except Exception:
            prob = float(preds)
        return {
            "feature_contributions": contributions,
            "predicted_churn": int(preds),
            "churn_probability": prob
        }
    except Exception:
        preds_df = predict_df(model, row_df)
        row = preds_df.iloc[0]
        return {
            "feature_contributions": None,
            "predicted_churn": int(row["predicted_churn"]),
            "churn_probability": float(row["churn_probability"])
        }

# ---------------------------
# Simulation with ROI
# ---------------------------
def simulate_action_with_roi(model: Any, df: pd.DataFrame, action: Dict[str, Any], cost_per_customer: float = 0.0) -> Dict[str, Any]:
    """
    Heuristic simulation for retention actions:
    - picks customers whose churn_probability >= target_threshold
    - reduces their churn_probability by a heuristic reduction factor based on discount_pct and extend_months
    - computes retained_customers, revenue_saved, action_cost and ROI
    """
    if not hasattr(model, "encoders"):
        raise ValueError("Model must have preprocessing artifacts (train with train_model first).")

    baseline_preds = predict_df(model, df.copy())
    before_churn_rate = float(baseline_preds["predicted_churn"].mean())
    threshold = float(action.get("target_threshold", 0.5))
    mask = baseline_preds["churn_probability"] >= threshold
    n_target = int(mask.sum())

    discount_pct = float(action.get("discount_pct", 0.0))
    extend_months = float(action.get("extend_months", 0.0))

    reduction_factor = 1.0 - (discount_pct * 0.005 + extend_months * 0.01)
    reduction_factor = max(0.0, min(1.0, reduction_factor))

    df_after = baseline_preds.copy()
    df_after.loc[mask, "churn_probability"] = df_after.loc[mask, "churn_probability"] * reduction_factor
    df_after["predicted_churn_after"] = (df_after["churn_probability"] >= threshold).astype(int)
    after_churn_rate = float(df_after["predicted_churn_after"].mean())

    retained_customers = int(round((before_churn_rate - after_churn_rate) * len(df)))
    arpc = float(action.get("avg_revenue_per_customer", 100.0))
    revenue_saved = float(round(retained_customers * arpc, 2))
    action_cost = float(round(n_target * float(cost_per_customer or 0.0), 2))
    if action_cost > 0:
        roi = float(round(((revenue_saved - action_cost) / action_cost) * 100.0, 2))
    else:
        roi = float(0.0)

    return {
        "before_churn_rate": before_churn_rate,
        "after_churn_rate": after_churn_rate,
        "retained_customers": retained_customers,
        "revenue_saved": revenue_saved,
        "action_cost": action_cost,
        "roi": roi,
        "n_targeted_customers": n_target,
        "reduction_factor": reduction_factor
    }

# End of core.py
