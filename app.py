"""
app.py

Notes / high-level:
- Sessions are in-memory and recovered automatically if user invokes endpoints before uploading.
- Automatic background session cleanup is disabled by default (you told me session TTL was annoying).
- Predict endpoint now returns a JSON preview (dashboard-friendly) when requested (Accept header or preview flag).
- Explain endpoint supports GET by row_index and POST single-row file upload.
- Small, clear comments for each section (so you remember why it's there).
"""

import time
import uuid
import traceback
from io import BytesIO
from datetime import datetime
from typing import Optional
import pandas as pd
from flask import Flask, request, render_template, jsonify, send_file, make_response

# local modules
import core
import chatbot

# -----------------------
# Flask app config
# -----------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['MAX_CONTENT_LENGTH'] = 250 * 1024 * 1024  # 250 MB
# ephemeral secret ok for local/dev
app.secret_key = uuid.uuid4().hex

# -----------------------
# In-memory sessions
# -----------------------
# NOTE: We keep sessions in memory. By your request, automatic cleanup is not enforced
# (we avoid surprising the user by removing sessions while they are analyzing).
# If you later want TTL-based cleanup re-enabled, you can re-enable the cleanup thread.

IN_MEMORY_SESSIONS = {}

# (left here for reference — TTL removed by default)
SESSION_TTL_SECONDS = None  # set to None to disable automatic expiry
CLEANUP_INTERVAL_SECONDS = 60  # used only if you re-enable cleanup

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
    }
    return sid

def get_session(sid: Optional[str]):
    """Return session by sid or None."""
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
    """Render SPA and provide a fresh session id."""
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

        # model_type selection (many options supported via core.get_model)
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

        # call core.train_model (returns model_id, meta, model_obj)
        model_id, meta, model_obj = core.train_model(
            df=df,
            target_col=target_col,
            model_type=model_type,
            sample_ratio=sample_ratio,
            mode=mode,
            compute_cv=compute_cv
        )

        # core returns metadata dict; handle errors
        if not meta or (isinstance(meta, dict) and meta.get("status") == "error"):
            msg = meta.get("error") if isinstance(meta, dict) else "training failed"
            return _json_error(f"training failed: {msg}", 400)

        # save model & metrics so prediction/explain/simulate can use it
        session["model"] = model_obj
        session["metrics"] = meta.get("metrics") if isinstance(meta, dict) else None

        # shap best-effort
        try:
            shap_summary = core.compute_shap_summary(model_obj, df, sample_limit=5000)
            session["shap"] = shap_summary
        except Exception:
            session["shap"] = None

        # small KPIs (churn_rate and n_rows) for dashboard
        kpis = {}
        try:
            y = df[target_col]
            if len(y.dropna()) > 0:
                mapped = y.map(lambda v: 1 if str(v).strip().lower() in ("1", "yes", "true", "y", "churn") else 0) \
                          if not (pd.api.types.is_numeric_dtype(y)) else y
                kpis["churn_rate"] = float(mapped.dropna().mean())
                kpis["n_rows"] = int(len(mapped.dropna()))
        except Exception:
            pass

        return jsonify({
            "model_id": model_id,
            "meta": meta,
            "shap": session["shap"],
            "kpis": kpis,
            "session_id": sid
        })
    except Exception as e:
        traceback.print_exc()
        return _json_error(f"training failed: {str(e)}", 500)

# Predict: returns JSON preview/dashboard info when requested (preview flag or Accept header),
# otherwise returns CSV download.
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

        # prefer uploaded file for scoring, else use session df
        df_pred = None
        if "file" in request.files:
            try:
                df_pred = core.safe_read_csv_bytes(request.files["file"].read())
            except Exception:
                df_pred = None
        if df_pred is None:
            df_pred = session.get("df")
        if df_pred is None:
            return _json_error("no data available for prediction")

        predictions_df = core.predict_df(model_obj, df_pred.copy())
        session["predictions"] = predictions_df

        # Determine preview mode:
        accept = request.headers.get("Accept", "")
        preview_flag = (request.form.get("preview") or request.args.get("preview") or "").lower() in ("1", "true", "yes", "y")
        # if browser requested JSON or preview flag, return dashboard-style JSON
        if "application/json" in accept or "text/html" in accept or preview_flag:
            preview_html = core.preview_df_html(predictions_df.head(200), max_rows=200, max_cols=50)
            churn_rate = float(predictions_df["predicted_churn"].mean())
            revenue_col = None
            for candidate in ("Revenue", "Total Spend", "ARPU", "ARPC", "TotalSpend", "Revenue_USD", "revenue"):
                if candidate in predictions_df.columns:
                    revenue_col = candidate
                    break
            revenue_summary = None
            if revenue_col:
                try:
                    revenue_summary = {
                        "avg_revenue": float(predictions_df[revenue_col].mean()),
                        "sum_revenue": float(predictions_df[revenue_col].sum())
                    }
                except Exception:
                    revenue_summary = None
            # simple KPI cards and sample rows for frontend dashboard
            return jsonify({
                "preview_html": preview_html,
                "n_rows": int(len(predictions_df)),
                "churn_rate": churn_rate,
                "revenue_summary": revenue_summary,
                "rows_sample": predictions_df.head(200).to_dict(orient="records"),
                "session_id": sid
            })

        # default behavior: return CSV for download
        return _make_csv_response(predictions_df)
    except Exception as e:
        traceback.print_exc()
        return _json_error(f"prediction failed: {str(e)}", 500)

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
        result = core.simulate_action_with_roi(model_obj, df, action, cost_per_customer)
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
# Clear session: explicit only (accept JSON, form, args or raw)
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
