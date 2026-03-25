#core.py 

""" These are Python utility modules. They help with type hints, file handling, warnings, error tracking, and time-based operations."""
from typing import Tuple, Dict, Any, Optional
import io
import warnings
import traceback
import time  

"""These libraries are used for reading datasets and numerical processing."""
import chardet
import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV #These functions are used to split data into training and testing sets and to evaluate models.
from sklearn.preprocessing import StandardScaler #This class is used for scaling features to have mean=0 and variance=1.
from sklearn.impute import SimpleImputer #This class is used for imputing missing values in a dataset.
from sklearn.linear_model import LogisticRegression #This class is used for implementing logistic regression models.
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier #These classes are used for implementing ensemble learning algorithms.
from sklearn.tree import DecisionTreeClassifier #This class is used for implementing decision tree algorithms.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score #These functions are used for evaluating the performance of machine learning models.
from sklearn.naive_bayes import GaussianNB #This class is used for implementing Gaussian Naive Bayes algorithm.

# shap import 
"""This attempts to import the shap library, which is used for SHapley Additive exPlanations. It also handles the case where the library is not available."""
try:
    import shap  
    SHAP_AVAILABLE = True
    warnings.filterwarnings("ignore", category=UserWarning)
except Exception:
    shap = None
    SHAP_AVAILABLE = False

# heavy libraries import 
try:
    from xgboost import XGBClassifier  #This attempts to import the XGBClassifier from the xgboost library, which is used for eXtreme Gradient Boosting. It also handles the case where the library is not available.
except Exception:
    XGBClassifier = None # This sets the XGBClassifier variable to None if the import fails. This is likely done to handle the case where the library is not available, and to prevent the program from crashing. If we don't set it to none, the program may raise an error when trying to use the XGBClassifier variable later on.

try:
    from catboost import CatBoostClassifier  #This attempts to import the CatBoostClassifier from the catboost library, which is used for CatBoost algorithm.
except Exception:
    CatBoostClassifier = None  # This sets the CatBoostClassifier variable to None if the import fails. This is likely done to handle the case where the library is not available, and to prevent the program from crashing. If we don't set it to none, the program may raise an error when trying to use the CatBoostClassifier variable later on.

# ---------------------------
# Global prediction threshold
# ---------------------------
# Lower than 0.5 to improve recall on imbalanced churn data
DEFAULT_THRESHOLD: float = 0.35

# ---------------------------
# Safe file reading (CSV/XLSX)
# ---------------------------
""" This function safely reads an uploaded file (CSV or Excel) and converts it into a pandas DataFrame without crashing due to encoding or format issues. """

def safe_read_csv_bytes(raw_bytes: bytes) -> Optional[pd.DataFrame]:
    try:
        if raw_bytes is None:
            return None

        # This block detects Excel files by checking their ZIP file signature. Since .xlsx files are internally ZIP archives, they start with the ‘PK’ signature. Excel files are read using pandas’ read_excel, which does not require manual encoding handling. If reading fails, the system falls back to CSV parsing.
        if isinstance(raw_bytes, (bytes, bytearray)) and raw_bytes[:4] == b'PK\x03\x04':
            try:
                df = pd.read_excel(io.BytesIO(raw_bytes))
                df._detected_encoding = "xlsx"
                return df
            except Exception:
                # fall back to CSV parsing
                pass

        # This uses the chardet library to estimate the CSV file’s encoding. The detected encoding is prioritized, and additional common encodings like UTF-8 and Latin-1 are added as fallbacks. This prepares the system to handle files from different operating systems and software sources.
        # The system first uses the chardet library to guess the encoding of the CSV file. That guessed encoding is tried first. If that doesn’t work, the system tries a list of common encodings as fallbacks
        guess = chardet.detect(raw_bytes or b"")
        detected = guess.get("encoding") or "" 
        candidates = [detected] if detected else [] # 
        candidates += ["utf-8-sig", "utf-8", "cp1252", "latin1"]  
        
        # If chardet fails to detect encoding, fallback encodings are tried one by one; if it detects an encoding, it is placed first in the candidate list and the loop tries encodings in order until one works. If all else fails, it returns None.
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
    info["n_rows"], info["n_cols"] = int(df.shape[0]), int(df.shape[1]) # Get the number of rows and columns in the DataFrame
    cols = [] 
    for c in df.columns:
        dtype = str(df[c].dtype)
        missing = int(df[c].isna().sum())
        cols.append({"name": c, "dtype": dtype, "missing": missing})
    info["columns"] = cols
    info["missing_total"] = int(df.isna().sum().sum()) # Calculate the total number of missing values in the DataFrame
    info["numeric_cols"] = [c for c in df.select_dtypes(include=[np.number]).columns.tolist()] # Get the list of numeric columns in the DataFrame
    info["categorical_cols"] = [c for c in df.columns if c not in info["numeric_cols"]] # Get the list of categorical columns in the DataFrame
    return info

def generate_full_report(df: pd.DataFrame, sample_limit: int = 50000) -> Dict[str, Any]:
    n = len(df)
    sample = df.sample(n=sample_limit, random_state=42) if n > sample_limit else df.copy() # Sample the data if it's larger than the limit
    report = quick_eda(sample) # Generate quick EDA report for the sampled data
    numeric_cols = report["numeric_cols"]
    if numeric_cols:
        try:
            report["correlation"] = sample[numeric_cols].corr().round(3).to_dict() # Calculate the correlation matrix for numeric columns and convert it to a dictionary
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
    return Xt

# ---------------------------
# Target variable encoder
# ---------------------------
def _encode_target(y_series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(y_series):
        return np.where(y_series > 0, 1, 0).astype(int)

    y_str = y_series.astype(str).str.lower().str.strip()
    
    positive_values = ['yes', 'true', '1', 'y', 'churn', 'positive']
    
    y_encoded = np.where(y_str.isin(positive_values), 1, 0)
    return y_encoded.astype(int)

# ---------------------------
# Preprocessing for model
# ---------------------------
def preprocess_for_model(df: pd.DataFrame, target_col: str, fit: bool = True,
                         encoders: Optional[Dict[str, Dict[Any, int]]] = None,
                         imputer: Optional[SimpleImputer] = None,
                         scaler: Optional[StandardScaler] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Dict[Any, int]], SimpleImputer, StandardScaler]:
    if target_col not in df.columns:
        raise ValueError(f"target_col {target_col} not in dataframe")

    dfc = df.copy()
    y = _encode_target(dfc[target_col])
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
_PARAM_GRIDS = {
    "logistic": {
        'C': [0.1, 1.0, 10]
    },
    "randomforest": {
        'n_estimators': [100, 150],
        'max_depth': [5, 10, None],
        'min_samples_leaf': [5, 10]
    },
    "decisiontree": {
        'max_depth': [3, 5, 7, 10],
        'min_samples_leaf': [5, 10, 20]
    },
    "gradientboost": {
        'n_estimators': [100, 150],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    },
    "xgboost": {
        'n_estimators': [100, 150],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    },
    "adaboost": {
        'n_estimators': [50, 100],
        'learning_rate': [0.1, 1.0]
    },
    "catboost": {
        'iterations': [200, 300],
        'depth': [4, 6],
        'learning_rate': [0.05, 0.1]
    },
    "naivebayes": {},
}

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
# Best metric selector
# ---------------------------
def select_best_metric(metrics: Dict[str, Any]) -> str:
    """Return the key of the most informative metric available.
    Priority: ROC-AUC > F1 > Accuracy.
    """
    if metrics.get("roc_auc") is not None:
        return "roc_auc"
    elif metrics.get("f1") is not None:
        return "f1"
    else:
        return "accuracy"


# ---------------------------
# Train model 
# ---------------------------
def train_model(df: pd.DataFrame,
                target_col: str,
                model_type: str = "logistic",
                sample_ratio: Optional[float] = 0.2,
                mode: str = "split",
                tune_model: bool = False,
                compute_cv: bool = False,
                cv_folds: int = 3,
                progress_callback: Optional[callable] = None) -> Tuple[str, Dict[str, Any], Any]:
    try:
        if progress_callback: # <--- PROGRESS REPORT 1
            progress_callback(5, "Preprocessing data...")
            time.sleep(0.5)

        if target_col not in df.columns:
            raise ValueError("target_col not found in dataframe")

        df_clean = df.dropna(subset=[target_col]).copy()
        if df_clean.shape[0] == 0:
            raise ValueError("no rows with non-null target to train on")

        X_all_scaled, y_all, encoders, imputer, scaler = preprocess_for_model(df_clean, target_col, fit=True)

        if progress_callback: # <--- PROGRESS REPORT 2
            progress_callback(20, "Scaling and splitting data...")
            time.sleep(0.5)

        if mode == "split":
            try:
                sample_ratio = float(sample_ratio)
            except Exception:
                sample_ratio = 0.2
            if not (0.0 < sample_ratio < 1.0):
                sample_ratio = 0.2
            
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
        
        else:
            sample_ratio = None
            X_train, y_train = X_all_scaled, y_all
            X_test, y_test = X_all_scaled, y_all

        estimator = get_model(model_type)
        

        if tune_model:
            if progress_callback: # <--- PROGRESS REPORT 3 (Tuning start)
                progress_callback(35, "Tuning model hyperparameters...")
                time.sleep(0.5)

            model_key = model_type.lower().replace(" ", "")
            param_grid = _PARAM_GRIDS.get(model_key, {})
            
            if param_grid:
                print(f"--- Tuning hyperparameters for {model_type} ---")
                
                grid_search = GridSearchCV(
                    estimator=estimator,
                    param_grid=param_grid,
                    cv=3,
                    n_jobs=-1,
                    scoring='accuracy'
                )
                
                grid_search.fit(X_train, y_train)
                estimator = grid_search.best_estimator_
                print(f"Best params found: {grid_search.best_params_}")
            else:
                print(f"No parameter grid for {model_type}, using defaults.")
                estimator.fit(X_train, y_train)

            if progress_callback: # <--- PROGRESS REPORT 4a (After Tuning/Before final fit)
                progress_callback(60, "Training final model...")
                time.sleep(0.5)
        else:
            if progress_callback: # <--- PROGRESS REPORT 4b (Non-tuning fit start)
                progress_callback(50, f"Training {model_type} model...")
                time.sleep(0.5)

            estimator.fit(X_train, y_train)

        if progress_callback: # <--- PROGRESS REPORT 5 (Metrics start)
            progress_callback(80, "Computing metrics...")
            time.sleep(0.5)

        # Use DEFAULT_THRESHOLD — consistent with predict_df() for accurate recall
        try:
            y_proba = estimator.predict_proba(X_test)  # full 2-col array for roc_auc
            y_pred  = (y_proba[:, 1] >= DEFAULT_THRESHOLD).astype(int)
        except Exception:
            y_proba = None
            y_pred  = estimator.predict(X_test)        # fallback if no predict_proba

        metrics = _compute_metrics(y_test, y_pred, y_proba)

        # ----------------------------------------------------------
        # Overfitting / underfitting detection
        # Compute train score using the same metric (prefer F1).
        # Only meaningful in split mode; full mode has no holdout.
        # ----------------------------------------------------------
        fit_status: str = "N/A"
        fit_reason: str = "No holdout set — trained on full data"
        train_score: float = 0.0
        test_score: float = 0.0

        if mode == "split":
            try:
                y_proba_tr = estimator.predict_proba(X_train)
                y_pred_tr  = (y_proba_tr[:, 1] >= DEFAULT_THRESHOLD).astype(int)
            except Exception:
                y_pred_tr = estimator.predict(X_train)

            train_metrics = _compute_metrics(y_train, y_pred_tr)

            # Prefer F1; fall back to accuracy
            train_score = float(
                train_metrics["f1"] if train_metrics.get("f1") is not None
                else train_metrics.get("accuracy", 0.0)
            )
            test_score = float(
                metrics["f1"] if metrics.get("f1") is not None
                else metrics.get("accuracy", 0.0)
            )

            gap = train_score - test_score

            if train_score < 0.6 and test_score < 0.6:
                fit_status = "Underfitting"
                fit_reason = "Model performs poorly on both training and test data"
            elif gap > 0.15:
                fit_status = "Severe Overfitting"
                fit_reason = "Large gap between training and test performance"
            elif gap > 0.08:
                fit_status = "Overfitting"
                fit_reason = "Model performs significantly better on training data"
            elif gap > 0.03:
                fit_status = "Mild Overfitting"
                fit_reason = "Small gap between training and test performance"
            elif test_score > 0.75:
                fit_status = "Good Fit"
                fit_reason = "Strong and balanced generalization"
            else:
                fit_status = "Acceptable"
                fit_reason = "Moderate performance with reasonable generalization"

        if compute_cv and mode == "split":
            try:
                cv_scores = cross_val_score(estimator, X_train, y_train, cv=min(cv_folds, 5), scoring="accuracy", n_jobs=-1)
                metrics["cv_mean"] = float(np.mean(cv_scores))
                metrics["cv_std"] = float(np.std(cv_scores))
            except Exception:
                pass
        
        model_obj = estimator
        model_obj.encoders = encoders
        model_obj.imputer = imputer
        model_obj.scaler = scaler
        model_obj.target_col = target_col
        # Store exact feature columns used at fit time — used to align predict/explain
        model_obj.feature_columns = df_clean.drop(columns=[target_col]).columns.tolist()

        model_id = f"{model_type}_{int(np.random.randint(1000, 9999))}"
        
        best_metric = select_best_metric(metrics)

        if mode == "split":
            meta = {
                "status": "success",
                "trained_on": "split",
                "test_size": float(sample_ratio or 0.2),
                "metrics": metrics,
                "best_metric": best_metric,
                "best_score": metrics.get(best_metric),
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "n_rows": int(len(y_train)),
                "train_score": round(train_score, 4),
                "test_score": round(test_score, 4),
                "fit_status": fit_status,
                "fit_reason": fit_reason,
            }
        else:
            meta = {
                "status": "success",
                "trained_on": "full",
                "metrics": metrics,
                "best_metric": best_metric,
                "best_score": metrics.get(best_metric),
                "n_rows": int(len(y_all)),
                "train_score": None,
                "test_score": None,
                "fit_status": "N/A",
                "fit_reason": "No holdout set — trained on full data",
            }
             
        if progress_callback: # <--- FINAL SUCCESS REPORT
            progress_callback(100, "Success.")

        return model_id, meta, model_obj

    except Exception as exc:
        traceback.print_exc()
        return "", {"status": "error", "error": str(exc)}, None

# ---------------------------
# Prediction
# ---------------------------
def predict_df(model: Any, df: pd.DataFrame) -> pd.DataFrame:
    if not hasattr(model, "encoders") or not hasattr(model, "imputer") or not hasattr(model, "scaler"):
        raise ValueError("Model missing preprocessing artifacts. Train model using train_model first.")

    df_copy = df.copy()
    if hasattr(model, "target_col") and model.target_col in df_copy.columns:
        df_proc = df_copy.drop(columns=[model.target_col])
    else:
        df_proc = df_copy

    # Align to training columns: drops unseen columns, fills missing ones with 0
    if hasattr(model, "feature_columns"):
        df_proc = df_proc.reindex(columns=model.feature_columns, fill_value=0)

    # Transform with the encoders (unknown -> -1)
    X_enc = _transform_with_encoders(df_proc, model.encoders or {})
    # Impute / scale (use model's artifacts)
    X_imputed = model.imputer.transform(X_enc)
    X_scaled = model.scaler.transform(X_imputed)

    # Predict
    try:
        preds = model.predict(X_scaled)
    except Exception:
        # fallback: all zeros
        preds = np.zeros((X_scaled.shape[0],), dtype=int)

    # Probability
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

    out = df_copy.copy()
    out["churn_probability"] = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    # Apply DEFAULT_THRESHOLD — consistent with training evaluation
    out["predicted_churn"] = (out["churn_probability"] >= DEFAULT_THRESHOLD).astype(int)

    return out

# ---------------------------
# SHAP summary 
# ---------------------------
def compute_shap_summary(model: Any, df: pd.DataFrame, sample_limit: int = 5000) -> Optional[Dict[str, Any]]:
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
        # Align to training columns: drops unseen columns, fills missing ones with 0
        if hasattr(model, "feature_columns"):
            df_proc = df_proc.reindex(columns=model.feature_columns, fill_value=0)
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