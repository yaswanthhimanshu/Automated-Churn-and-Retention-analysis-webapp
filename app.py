#  app.py


import time
import uuid
import traceback
from io import BytesIO
from datetime import datetime
from typing import Optional
import pandas as pd
from flask import Flask, request, render_template, jsonify, send_file, make_response
import threading

# local modules
import core
import chatbot
import time_churn_analysis

from dotenv import load_dotenv
load_dotenv()


# -----------------------
# Flask app config
# -----------------------
# ===========================================================================
# JSON serialization  —  production-grade, handles all pandas / numpy types
# ===========================================================================
import math as _math
import numpy as _np
import pandas as _pd
from flask.json.provider import DefaultJSONProvider


class _SafeJSONProvider(DefaultJSONProvider):
    """
    Replaces Flask's default JSON provider globally.
    Sanitizes the full object tree before json.dumps so NaN/Inf/numpy/pandas
    types never produce invalid JSON.

    Conversions applied:
      float NaN / Inf / -Inf          -> null
      np.floating (any width)         -> float  (NaN/Inf -> null)
      np.integer  (any width + uint)  -> int
      np.bool_                        -> bool
      np.str_                         -> str
      np.ndarray                      -> list   (recursive)
      pd.Series                       -> list   (recursive)
      pd.DataFrame                    -> list-of-dicts  (records orient)
      pd.Timestamp                    -> ISO-8601 string
      pd.NA / pd.NaT                  -> null
      dict                            -> recurse values; keys coerced to str
      list / tuple                    -> recurse elements
      anything else                   -> str() fallback, never raises
    """
    _np_float    = (_np.floating,)
    _np_int      = (_np.integer,)
    _np_bool     = (_np.bool_,)
    _np_str      = (_np.str_,)
    _np_ndarray  = _np.ndarray
    _pd_series   = _pd.Series
    _pd_df       = _pd.DataFrame
    _pd_ts       = _pd.Timestamp
    _pd_na_type  = type(_pd.NA)
    _pd_nat_type = type(_pd.NaT)

    @classmethod
    def _sanitize(cls, obj):
        if obj is None:
            return None
        if isinstance(obj, (cls._pd_na_type, cls._pd_nat_type)):
            return None
        if isinstance(obj, cls._pd_ts):
            return obj.isoformat()
        if isinstance(obj, cls._pd_df):
            return cls._sanitize(obj.to_dict(orient="records"))
        if isinstance(obj, cls._pd_series):
            return cls._sanitize(obj.tolist())
        if isinstance(obj, cls._np_ndarray):
            return cls._sanitize(obj.tolist())
        if isinstance(obj, cls._np_float):
            v = float(obj)
            return None if (_math.isnan(v) or _math.isinf(v)) else v
        if isinstance(obj, cls._np_int):
            return int(obj)
        if isinstance(obj, cls._np_bool):
            return bool(obj)
        if isinstance(obj, cls._np_str):
            return str(obj)
        if isinstance(obj, float):
            return None if (_math.isnan(obj) or _math.isinf(obj)) else obj
        if isinstance(obj, (int, str, bool)):
            return obj
        if isinstance(obj, dict):
            return {str(k): cls._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [cls._sanitize(v) for v in obj]
        try:
            return str(obj)
        except Exception:
            return None

    def dumps(self, obj, **kwargs):
        safe = self._sanitize(obj)
        kwargs.setdefault("ensure_ascii", self.ensure_ascii)
        kwargs.setdefault("sort_keys", self.sort_keys)
        kwargs["allow_nan"] = False
        return __import__("json").dumps(safe, **kwargs)


# ---------------------------------------------------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")

# Register safe JSON provider (must be immediately after Flask() init)
app.json_provider_class = _SafeJSONProvider
app.json = _SafeJSONProvider(app)
app.config['MAX_CONTENT_LENGTH'] = 250 * 1024 * 1024  # 250 MB
app.secret_key = uuid.uuid4().hex

# -----------------------
# In-memory sessions
# -----------------------

IN_MEMORY_SESSIONS = {}

SESSION_TTL_SECONDS = None  
CLEANUP_INTERVAL_SECONDS = 60 

def new_session():
    """Create a fresh session dictionary and return sid."""
    sid = uuid.uuid4().hex[:12]
    now_ts = time.time()
    IN_MEMORY_SESSIONS[sid] = {
        "df": None,
        "eda": None,
        "eda_full": None,
        "model": None,
        "metrics": None,
        "predictions": None,
        "shap": None,
        "simulate_result": None,
        "created_ts": now_ts,
        "uploaded_at": None,
        "encoding": None,

        "train_progress": 0,
        "train_status_msg": "Idle",
        "model_history": [],      # stores last 5 trained model metric summaries
    }
    return sid

def get_session(sid: Optional[str]):
    """Returns session by sid or None."""
    if not sid:
        return None
    return IN_MEMORY_SESSIONS.get(sid)

def remove_session(sid: str):
    """Explicitly delete a session (called when user asks to clear)."""
    if sid in IN_MEMORY_SESSIONS:
        try:
            del IN_MEMORY_SESSIONS[sid]
        except Exception:
            pass

# -----------------------
# Session recovery helper
# -----------------------
def ensure_session(sid: Optional[str] = None):
    """
    If sid is valid return (sid, session).
    If not valid or missing, create a new session and return it.
    This prevents 'invalid session' errors when users click features before upload.
    """
    if sid:
        s = get_session(sid)
        if s is not None:
            return sid, s
    ns = new_session()
    return ns, get_session(ns)


# -----------------------
# Helpers (responses)
# -----------------------
def _json_error(msg, code=400):
    return jsonify({"error": msg}), code

def _make_csv_response(df, filename="predictions.csv"):
    """Return a downloadable CSV response from pandas DataFrame."""
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    mem = BytesIO(csv_bytes)
    mem.seek(0)
    resp = make_response(send_file(mem, as_attachment=True, download_name=filename, mimetype="text/csv"))
    resp.headers["Cache-Control"] = "no-store, no-cache, private"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

# -----------------------
# Routes
# -----------------------

@app.route("/")
def index():
    sid = new_session()
    return render_template("index.html", session_id=sid)

@app.route("/upload", methods=["POST"])
def upload():
    try:
        sid_in = request.form.get("session_id") or request.args.get("session_id")
        sid, session = ensure_session(sid_in)

        file = request.files.get("file")
        if not file:
            return _json_error("no file uploaded")
        raw = file.read()
        df = core.safe_read_csv_bytes(raw)
        if df is None:
            return _json_error("failed to parse uploaded file (unsupported format or encoding)")

        # Save dataset and metadata to session
        session["df"] = df
        session["uploaded_at"] = datetime.utcnow().isoformat() + "Z"
        session["created_ts"] = time.time()
        session["encoding"] = getattr(df, "_detected_encoding", None)

        # Build quick preview + EDA for frontend dashboard
        preview_html = core.preview_df_html(df, max_rows=1000, max_cols=1000)
        eda_summary = core.quick_eda(df)
        full_report = core.generate_full_report(df, sample_limit=2000)

        cols = df.columns.tolist()
        likely_targets = [c for c in cols if any(k in c.lower() for k in ("churn", "target", "exit", "exited", "label"))]

        # reset model/predictions when new dataset is uploaded
        session.update({
            "eda": eda_summary,
            "eda_full": full_report,
            "model": None,
            "metrics": None,
            "predictions": None,
            "shap": None,
            "simulate_result": None
        })

        return jsonify({
            "preview_html": preview_html,
            "eda": eda_summary,
            "eda_full": full_report,
            "columns": cols,
            "likely_targets": likely_targets,
            "session_id": sid
        })
    except Exception as e:
        traceback.print_exc()
        return _json_error(f"upload failed: {str(e)}", 500)

# Full EDA (recompute)
@app.route("/eda_full", methods=["POST"])
def eda_full():
    try:
        sid_in = request.form.get("session_id") or request.args.get("session_id")
        sid, session = ensure_session(sid_in)
        if not session or session.get("df") is None:
            return _json_error("invalid session or no dataset")
        report = core.generate_full_report(session["df"])
        session["eda_full"] = report
        return jsonify({"eda_full": report, "session_id": sid})
    except Exception as e:
        traceback.print_exc()
        return _json_error(f"full EDA failed: {str(e)}", 500)

# Train model - accepts dataset in request or uses session df.
@app.route("/train", methods=["POST"])
def train():
    try:
        sid_in = request.form.get("session_id") or request.args.get("session_id")
        sid, session = ensure_session(sid_in)

        # If user provided a file in train request, accept it (upload-with-train flow)
        if session.get("df") is None and "file" in request.files:
            raw = request.files["file"].read()
            df_new = core.safe_read_csv_bytes(raw)
            if df_new is None:
                return _json_error("failed to parse uploaded file for training")
            session["df"] = df_new
            session["uploaded_at"] = datetime.utcnow().isoformat() + "Z"
            session["encoding"] = getattr(df_new, "_detected_encoding", None)

        if not session or session.get("df") is None:
            return _json_error("invalid session or no dataset")

        df = session["df"]
        target_col = request.form.get("target_col")
        if not target_col or target_col not in df.columns:
            return _json_error("invalid or missing target column")

        if session.get("train_progress", 0) > 0 and session.get("train_progress", 0) < 100:
            return _json_error("Training is already running.", 409)

        session.update({
            "train_progress": 0,
            "train_status_msg": "Queued",
            "model": None,
            "metrics": None,
            "predictions": None,
            "shap": None,
            "simulate_result": None
        })
        
        # model_type selection 
        model_type = request.form.get("model_type", "logistic").strip() or "logistic"

        # mode (full or split)
        mode_raw = (request.form.get("mode") or "fast").lower()
        mode = "full" if mode_raw in ("full", "train_full", "all") else "split"

        # sample_ratio handling for split mode
        if mode == "split":
            try:
                sample_ratio = float(request.form.get("sample_ratio", 0.2))
            except Exception:
                sample_ratio = 0.2
            if not (0.0 < sample_ratio < 1.0):
                sample_ratio = 0.2
        else:
            sample_ratio = None

        compute_cv = (request.form.get("compute_cv") or "false").lower() in ("1", "true", "yes", "y")
        tune_model = (request.form.get("tune_model") or "false").lower() in ("1", "true", "yes", "y")

        # --- Column selection: drop user-chosen columns ONLY from the training copy ---
        # Original session df is NEVER modified — EDA/chat/analysis/lifecycle stay intact.
        raw_drop = request.form.get("columns_to_drop", "")
        columns_to_drop = [c.strip() for c in raw_drop.split(",") if c.strip()] if raw_drop.strip() else []
        columns_to_drop = [c for c in columns_to_drop if c != target_col]  # never drop target
        df_train = df.drop(columns=columns_to_drop, errors="ignore").copy()
        session["columns_to_drop"] = columns_to_drop  # saved so /predict can align features


        def train_task(sid, df, target_col, model_type, sample_ratio, mode, compute_cv, tune_model):
            """Wrapper function to run the training and update session."""
            
            def progress_callback(progress_pct, msg):
                s = get_session(sid)
                if s:
                    s["train_progress"] = progress_pct
                    s["train_status_msg"] = msg
                    
            try:
                progress_callback(1, "Starting training thread...")
                
                # call core.train_model (returns model_id, meta, model_obj)
                model_id, meta, model_obj = core.train_model(
                    df=df, target_col=target_col, model_type=model_type,
                    sample_ratio=sample_ratio, mode=mode,
                    compute_cv=compute_cv, tune_model=tune_model,
                    progress_callback=progress_callback
                )
                
                s = get_session(sid)
                if not s:
                    return

                # core returns metadata dict; handle errors
                if not meta or (isinstance(meta, dict) and meta.get("status") == "error"):
                    msg = meta.get("error") if isinstance(meta, dict) else "training failed"
                    progress_callback(-1, f"Training failed: {msg}")
                    return

                # save model & metrics so prediction/explain/simulate can use it
                s["model"] = model_obj
                s["metrics"] = meta.get("metrics") if isinstance(meta, dict) else None
                s["meta"] = meta

                # Append to model history (last 5 only, metrics only)
                _m = meta.get("metrics") or {}
                _entry = {
                    "model_type": model_type,
                    "accuracy":   round(float(_m.get("accuracy",  0) or 0), 4),
                    "precision":  round(float(_m.get("precision", 0) or 0), 4),
                    "recall":     round(float(_m.get("recall",    0) or 0), 4),
                    "f1":         round(float(_m.get("f1",        0) or 0), 4),
                    "roc_auc":    round(float(_m["roc_auc"]), 4) if _m.get("roc_auc") is not None else None,
                    "trained_on": meta.get("trained_on", ""),
                    "n_rows":     meta.get("n_rows"),
                    "train_score": round(float(meta.get("train_score") or 0), 4) if meta.get("train_score") is not None else None,
                    "test_score":  round(float(meta.get("test_score")  or 0), 4) if meta.get("test_score")  is not None else None,
                    "fit_status":  meta.get("fit_status", ""),
                    "fit_reason":  meta.get("fit_reason", ""),
                }
                history = s.get("model_history") or []
                history.append(_entry)
                s["model_history"] = history[-5:]
                
                progress_callback(95, "Computing SHAP summary...")
                time.sleep(0.5)
                
                # shap best-effort
                try:
                    shap_summary = core.compute_shap_summary(model_obj, df, sample_limit=5000)
                    s["shap"] = shap_summary
                except Exception:
                    s["shap"] = None
                
                if s["train_progress"] < 100:
                     progress_callback(100, "Success.")

            except Exception as e:
                s = get_session(sid)
                if s:
                    s["train_progress"] = -1
                    s["train_status_msg"] = f"Training error: {str(e)}"
                traceback.print_exc()

        thread = threading.Thread(
            target=train_task, 
            args=(sid, df_train, target_col, model_type, sample_ratio, mode, compute_cv, tune_model)
        )
        thread.start()
        
        return jsonify({
            "status": "training_started",
            "message": "Training started asynchronously. Poll /train_status for progress.",
            "session_id": sid,
            "progress": 0
        }), 202 
    except Exception as e:
        traceback.print_exc()
        return _json_error(f"training failed to start: {str(e)}", 500)

# Predict: returns  preview/dashboard info when requested 
@app.route("/predict", methods=["POST"])
def predict():
    try:
        sid_in = request.form.get("session_id") or request.args.get("session_id")
        sid, session = ensure_session(sid_in)
        if not session:
            return _json_error("invalid session")

        model_obj = session.get("model")
        if not model_obj:
            return _json_error("no trained model")

        # prefers uploaded file for scoring, else use session df
        df_raw = None
        if "file" in request.files:
            try:
                df_raw = core.safe_read_csv_bytes(request.files["file"].read())
            except Exception:
                df_raw = None
        if df_raw is None:
            df_raw = session.get("df")
        if df_raw is None:
            return _json_error("no data available for prediction")

        # Keep original df intact; drop identifier columns only for model input
        df_original = df_raw.copy()
        _cols_drop = session.get("columns_to_drop") or []
        df_for_model = df_raw.drop(columns=_cols_drop, errors="ignore").copy() if _cols_drop else df_raw.copy()

        # Run predictions on the feature-aligned copy
        pred_out = core.predict_df(model_obj, df_for_model)

        # Attach predictions back to the ORIGINAL dataframe (all original columns preserved)
        df_original["churn_probability"] = pred_out["churn_probability"].values
        df_original["predicted_churn"]   = pred_out["predicted_churn"].values

        session["predictions"] = df_original

        # Risk segment counts from FULL dataset
        probs = df_original["churn_probability"]
        risk_counts = {
            "high":   int((probs >= 0.7).sum()),
            "medium": int(((probs >= 0.35) & (probs < 0.7)).sum()),
            "low":    int((probs < 0.35).sum()),
        }

        # Determined preview mode:
        accept = request.headers.get("Accept", "")
        preview_flag = (request.form.get("preview") or request.args.get("preview") or "").lower() in ("1", "true", "yes", "y")
        if "application/json" in accept or "text/html" in accept or preview_flag:
            churn_rate = float(df_original["predicted_churn"].mean())
            revenue_col = None
            for candidate in ("Revenue", "Total Spend", "ARPU", "ARPC", "TotalSpend", "Revenue_USD", "revenue"):
                if candidate in df_original.columns:
                    revenue_col = candidate
                    break
            revenue_summary = None
            if revenue_col:
                try:
                    revenue_summary = {
                        "avg_revenue": float(df_original[revenue_col].mean()),
                        "sum_revenue": float(df_original[revenue_col].sum())
                    }
                except Exception:
                    revenue_summary = None

            # Top 100 rows sorted by churn_probability for the UI preview
            rows_sample = (
                df_original
                .sort_values("churn_probability", ascending=False)
                .head(100)
                .to_dict(orient="records")
            )
            # Full dataset — ALL rows in original order for download and analysis
            full_data = df_original.to_dict(orient="records")

            return jsonify({
                "n_rows": int(len(df_original)),
                "churn_rate": churn_rate,
                "revenue_summary": revenue_summary,
                "risk_counts": risk_counts,
                "rows_sample": rows_sample,
                "full_data": full_data,
                "session_id": sid
            })
        return _make_csv_response(df_original)
    except Exception as e:
        traceback.print_exc()
        return _json_error(f"prediction failed: {str(e)}", 500)

@app.route("/time_churn_detect", methods=["POST"])
def time_churn_detect():
    try:
        sid_in = request.form.get("session_id") or request.args.get("session_id")
        sid, session = ensure_session(sid_in)

        if not session or session.get("df") is None:
            return _json_error("invalid session or no dataset")

        df = session["df"]

        candidates = time_churn_analysis.detect_time_candidates(df)

        return jsonify({
            "candidates": candidates,
            "session_id": sid
        })

    except Exception as e:
        traceback.print_exc()
        return _json_error(f"time churn detection failed: {str(e)}", 500)

@app.route("/get_unique_values", methods=["POST"])
def get_unique_values():
    try:
        sid_in = request.form.get("session_id") or request.args.get("session_id")
        sid, session = ensure_session(sid_in)

        if not session or session.get("df") is None:
            return _json_error("invalid session or no dataset")

        df = session["df"]
        column = request.form.get("column")

        if not column:
            return _json_error("missing column")

        values = time_churn_analysis.get_column_unique_values(df, column)

        return jsonify({
            "column": column,
            "values": values,
            "session_id": sid
        })

    except Exception as e:
        traceback.print_exc()
        return _json_error(f"failed to get unique values: {str(e)}", 500)


@app.route("/time_churn", methods=["POST"])
def time_churn():
    try:
        sid_in = request.form.get("session_id") or request.args.get("session_id")
        sid, session = ensure_session(sid_in)

        if not session or session.get("df") is None:
            return _json_error("invalid session or no dataset")

        df = session["df"]

        time_column = request.form.get("time_column")
        analysis_type = request.form.get("analysis_type")

        if not time_column:
            return _json_error("missing time_column")

        user_value = None

        if pd.api.types.is_numeric_dtype(df[time_column]):

            if analysis_type == "range":
                min_val = request.form.get("min_value")
                max_val = request.form.get("max_value")

                try:
                    min_val = float(min_val)
                    max_val = float(max_val)
                except Exception:
                    return _json_error("invalid range values")

                if min_val > max_val:
                    return _json_error("min_value must be smaller than max_value")

                user_value = [min_val, max_val]

            else:
                val = request.form.get("user_value")

                try:
                    user_value = float(val)
                except Exception:
                    return _json_error("invalid numeric value")

        else:
            val = request.form.get("user_value")

            if not val:
                return _json_error("missing value for selected column")

            user_value = [v.strip() for v in val.split(",") if v.strip()]

        target_col = None

        if session.get("model") and hasattr(session["model"], "target_col"):
            target_col = session["model"].target_col
        else:
            return _json_error("target column not available (train model first)")

        predictions_df = session.get("predictions")
        if pd.api.types.is_numeric_dtype(df[time_column]):
            inferred_type = "duration"
        else:
            inferred_type = "period"

        result = time_churn_analysis.analyze_time_churn(
            df=df,
            target_col=target_col,
            time_column=time_column,
            time_type=inferred_type,
            user_value=user_value,
            predictions_df=predictions_df
        )

        lifecycle = time_churn_analysis.lifecycle_risk_analysis(
            df=df,
            target_col=target_col,
            predictions_df=predictions_df
        )
        session["lifecycle_risk"] = lifecycle

        return jsonify({
            "time_churn": result,
            "lifecycle_risk": lifecycle,
            "session_id": sid
        })

    except Exception as e:
        traceback.print_exc()
        return _json_error(f"time churn analysis failed: {str(e)}", 500)

@app.route("/lifecycle_risk", methods=["POST"])
def lifecycle_risk():
    try:
        sid_in = request.form.get("session_id") or request.args.get("session_id")
        sid, session = ensure_session(sid_in)

        if not session or session.get("df") is None:
            return _json_error("invalid session or no dataset")

        df = session["df"]

        # Get target column from trained model
        target_col = None
        if session.get("model") and hasattr(session["model"], "target_col"):
            target_col = session["model"].target_col
        else:
            return _json_error("target column not available (train model first)")

        predictions_df = session.get("predictions")

        result = time_churn_analysis.lifecycle_risk_analysis(
            df=df,
            target_col=target_col,
            predictions_df=predictions_df
        )

        if result is None:
            return _json_error("No duration-like column detected")

        session["lifecycle_risk"] = result

        return jsonify({
            "lifecycle_risk": result,
            "session_id": sid
        })

    except Exception as e:
        traceback.print_exc()
        return _json_error(f"lifecycle risk analysis failed: {str(e)}", 500)

    
# -----------------------
# Model History
# -----------------------
@app.route("/model_history", methods=["GET"])
def model_history():
    sid_in = request.args.get("session_id")
    sid, session = ensure_session(sid_in)
    return jsonify({
        "history": session.get("model_history") or [],
        "session_id": sid
    })

# -----------------------
# Training Status Route 
# -----------------------
@app.route("/train_status", methods=["GET"])
def train_status():
    sid_in = request.args.get("session_id")
    sid, session = ensure_session(sid_in)
    
    if not session:
        return _json_error("invalid session", 404)

    progress = session.get("train_progress", 0)
    msg = session.get("train_status_msg", "Idle")

    if progress >= 100 or progress == -1:
        meta = session.get("meta", {})
        
        kpis = {}
        if progress >= 100 and session.get("df") is not None:
            try:
                df = session["df"]
                target_col = None
                if hasattr(session.get("model"), 'target_col'):
                    target_col = session["model"].target_col
                
                if target_col and target_col in df.columns:
                    y = df[target_col]
                    if len(y.dropna()) > 0:
                        mapped = y.map(lambda v: 1 if str(v).strip().lower() in ("1", "yes", "true", "y", "churn") else 0) \
                                if not (pd.api.types.is_numeric_dtype(y)) else y
                        kpis["churn_rate"] = float(mapped.dropna().mean())
            except Exception:
                pass 

        if "n_rows" not in meta:
             if meta.get("trained_on") == "split":
                 meta["n_rows"] = meta.get("n_train")
             else:
                 meta["n_rows"] = session.get("df").shape[0] if session.get("df") is not None else 0
        
        return jsonify({
            "status": "completed" if progress >= 100 else "failed",
            "progress": progress,
            "message": msg,
            "meta": meta,
            "shap": session.get("shap"),
            "kpis": kpis,
            "session_id": sid
        })

    return jsonify({
        "status": "in_progress",
        "progress": progress,
        "message": msg,
        "session_id": sid
    })

# -----------------------
# Explain (per-row) route
# Supports GET (row_index) and POST (single-row CSV upload).
# -----------------------
@app.route("/explain", methods=["GET", "POST"])
def explain():
    try:
        if request.method == "GET":
            sid_in = request.args.get("session_id")
            row_index = request.args.get("row_index")
            sid, session = ensure_session(sid_in)
            if not session or session.get("model") is None or session.get("df") is None:
                return _json_error("missing session, model, or dataset for explanation")
            if row_index is None:
                return _json_error("missing row_index for GET explain")
            try:
                idx = int(row_index)
            except Exception:
                return _json_error("invalid row_index")
            df = session["df"]
            if idx < 0 or idx >= len(df):
                return _json_error("row_index out of range")
            row_df = df.iloc[[idx]].copy()
            explanation = core.explain_row(session["model"], row_df)
            return jsonify({"explanation": explanation, "session_id": sid})

        # POST path - file upload single-row
        sid_in = request.form.get("session_id") or request.args.get("session_id")
        sid, session = ensure_session(sid_in)
        if session is None or session.get("model") is None:
            return _json_error("missing session or trained model for explanation")
        if "file" not in request.files:
            return _json_error("no file provided for explanation")
        raw = request.files["file"].read()
        df_single = core.safe_read_csv_bytes(raw)
        if df_single is None or len(df_single) == 0:
            return _json_error("failed to parse uploaded single-row file")
        single = df_single.iloc[[0]].copy()
        explanation = core.explain_row(session["model"], single)
        return jsonify({"explanation": explanation, "session_id": sid})
    except Exception as e:
        traceback.print_exc()
        return _json_error(f"explain failed: {str(e)}", 500)

# -----------------------
# Simulate
# -----------------------
@app.route("/simulate", methods=["POST"])
def simulate():
    try:
        sid_in = request.form.get("session_id") or request.args.get("session_id")
        sid, session = ensure_session(sid_in)
        if not session:
            return _json_error("invalid session")

        df = session.get("df")
        model_obj = session.get("model")
        if df is None or model_obj is None:
            return _json_error("missing data or model")

        action = {}
        for key in ("discount_pct", "extend_months", "target_threshold"):
            val = request.form.get(key)
            if val is not None and val != "":
                try:
                    action[key] = float(val)
                except ValueError:
                    return _json_error(f"invalid {key}")

        cost_per_customer = float(request.form.get("cost_per_customer", 0))
        # Drop columns excluded at training time so features match the model
        _sim_drop = session.get("columns_to_drop") or []
        df_sim = df.drop(columns=_sim_drop, errors="ignore").copy() if _sim_drop else df
        result = core.simulate_action_with_roi(model_obj, df_sim, action, cost_per_customer)
        session["simulate_result"] = result
        return jsonify({"simulate": result, "session_id": sid})
    except Exception as e:
        traceback.print_exc()
        return _json_error(f"simulation failed: {str(e)}", 500)

# -----------------------
# Chat: use context from session
# -----------------------
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(silent=True) or {}
        sid_in = data.get("session_id") or request.form.get("session_id") or request.args.get("session_id")
        sid, session = ensure_session(sid_in)

        query = data.get("query", "")
        use_llm = bool(data.get("use_llm", False))

        context = {
            "eda": session.get("eda"),
            "metrics": session.get("metrics"),
            "shap": session.get("shap"),
            "simulate": session.get("simulate_result"),
        }
        answer = chatbot.respond_to_query(query, context, use_llm=use_llm)
        return jsonify({"answer": answer, "session_id": sid})
    except Exception as e:
        traceback.print_exc()
        return _json_error(f"chatbot error: {str(e)}", 500)

# -----------------------
# Clear session:
# -----------------------
@app.route("/clear_session", methods=["POST"])
def clear_session():
    try:
        sid = None
        data = request.get_json(silent=True)
        if isinstance(data, dict):
            sid = data.get("session_id") or data.get("sid")
        if not sid:
            sid = request.form.get("session_id") or request.args.get("session_id")
        if not sid:
            raw = request.get_data(as_text=True)
            if raw:
                try:
                    import json
                    parsed = json.loads(raw)
                    sid = parsed.get("session_id") or parsed.get("sid")
                except Exception:
                    sid = raw.strip() if raw.strip() else None
        if not sid:
            return _json_error("missing session_id")
        remove_session(sid)
        return jsonify({"ok": True})
    except Exception as e:
        traceback.print_exc()
        return _json_error(f"clear_session failed: {str(e)}", 500)

@app.route("/health")
def health():
    return jsonify({"status": "ok", "active_sessions": len(IN_MEMORY_SESSIONS)})

# -----------------------
# Run (development)
# -----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)