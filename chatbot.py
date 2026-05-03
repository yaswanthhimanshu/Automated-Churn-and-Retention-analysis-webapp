#chatbot.py
from typing import Dict, Any, List, Tuple
import re
import difflib
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False
# -------------------------
# Small formatters & helpers
# -------------------------
def _normalize(text: str) -> str:
    return (text or "").strip().lower()
def _shortlist_top_features_from_shap(shap: Dict[str, Any], top_n: int = 5) -> List[Dict[str, Any]]:
    if not shap or not isinstance(shap, dict):
        return []
    feats = shap.get("top_features") or shap.get("features") or []
    if not isinstance(feats, list):
        return []
    try:
        sorted_feats = sorted(feats, key=lambda f: float(f.get("mean_abs_shap", 0.0)), reverse=True)
    except Exception:
        sorted_feats = feats
    return sorted_feats[:top_n]
def _format_feature_list(features: List[Dict[str, Any]]) -> str:
    if not features:
        return "No feature analysis is available yet."
    lines = []
    for i, f in enumerate(features):
        name = f.get("name", "<unknown>") if isinstance(f, dict) else str(f)
        val = f.get("mean_abs_shap") if isinstance(f, dict) else None
        if val is not None:
            try:
                lines.append(f"- {name} (influence score: {float(val):.4f})")
            except Exception:
                lines.append(f"- {name} (influence score: {val})")
        else:
            lines.append(f"- {name}")
    return "\n".join(lines)
def _format_metrics(metrics: Dict[str, Any]) -> str:
    if not metrics:
        return "No model results available yet. Train a model to see performance."
    parts = []
    for k in ("roc_auc", "accuracy", "precision", "recall", "f1"):
        if k in metrics:
            try:
                parts.append(f"{k.upper()}: {float(metrics[k]):.3f}")
            except Exception:
                parts.append(f"{k}: {metrics[k]}")
    if parts:
        return ", ".join(parts)
    try:
        return ", ".join([f"{k}: {v}" for k, v in metrics.items()])
    except Exception:
        return "Metrics available but cannot format."
def _format_simulation(sim: Dict[str, Any]) -> str:
    if not sim:
        return "No simulation has been run yet. Head to the Simulate section to estimate the impact of a retention campaign."
    lines = []
    before = sim.get("before_churn_rate")
    after = sim.get("after_churn_rate")
    retained = sim.get("retained_customers")
    revenue = sim.get("revenue_saved")
    cost = sim.get("action_cost")
    roi = sim.get("roi")
    if before is not None:
        try:
            lines.append(f"Churn rate before campaign: {float(before):.1%}")
        except Exception:
            lines.append(f"Churn rate before campaign: {before}")
    if after is not None:
        try:
            lines.append(f"Churn rate after campaign: {float(after):.1%}")
        except Exception:
            lines.append(f"Churn rate after campaign: {after}")
    if retained is not None:
        try:
            lines.append(f"Customers estimated to be retained: {int(retained):,}")
        except Exception:
            lines.append(f"Customers estimated to be retained: {retained}")
    if revenue is not None:
        try:
            lines.append(f"Estimated revenue saved: ${float(revenue):,.2f}")
        except Exception:
            lines.append(f"Estimated revenue saved: {revenue}")
    if cost is not None:
        try:
            lines.append(f"Estimated campaign cost: ${float(cost):,.2f}")
        except Exception:
            lines.append(f"Estimated campaign cost: {cost}")
    if roi is not None:
        try:
            roi_val = float(roi)
            verdict = "This campaign is projected to be profitable." if roi_val > 0 else "This campaign may not be cost-effective at these settings."
            lines.append(f"Return on investment (ROI): {roi_val:.2f}%  —  {verdict}")
        except Exception:
            lines.append(f"Return on investment (ROI): {roi}")
    return "\n".join(lines) if lines else "Simulation result contained no recognizable fields."
def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None
def _as_rate(value: Any) -> float | None:
    val = _as_float(value)
    if val is None:
        return None
    return val / 100.0 if val > 1 else val
def _fmt_pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "not available"
    return f"{value * 100:.{digits}f}%"
def _get_actual_churn_rate(context: Dict[str, Any]) -> float | None:
    metrics = context.get("metrics") or {}
    if isinstance(metrics, dict) and "churn_rate" in metrics:
        return _as_rate(metrics.get("churn_rate"))
    return None
def _get_predicted_churn_rate(context: Dict[str, Any]) -> float | None:
    predictions = context.get("predictions") or {}
    if not isinstance(predictions, dict):
        return None
    if "predicted_churn_rate" in predictions:
        return _as_rate(predictions.get("predicted_churn_rate"))
    # Some older sessions used churn_rate inside the prediction summary. Treat it
    # only as predicted output, never as the historical/actual churn rate.
    if "churn_rate" in predictions:
        return _as_rate(predictions.get("churn_rate"))
    return None
def _get_model_type(context: Dict[str, Any]) -> str:
    metrics = context.get("metrics") or {}
    return str(
        context.get("current_model")
        or (metrics.get("model_type") if isinstance(metrics, dict) else "")
        or context.get("model_type")
        or ""
    ).strip()
def _get_metric(context: Dict[str, Any], key: str) -> float | None:
    metrics = context.get("metrics") or {}
    if isinstance(metrics, dict) and key in metrics:
        return _as_float(metrics.get(key))
    return None
def _fmt_metric_pct(context: Dict[str, Any], key: str) -> str:
    val = _get_metric(context, key)
    return _fmt_pct(_as_rate(val), 1) if val is not None else "not available"
def _recall_verdict(recall: Any) -> str:
    rate = _as_rate(recall)
    if rate is None:
        return "Recall is not available yet, so I cannot judge how many real churners the model is catching."
    if rate < 0.50:
        return "This model is missing many churners. It is not suitable for churn decisions in its current state."
    if rate < 0.70:
        return "This model captures most churners, but recall can still improve."
    return "This model is strong at detecting churners."
def _recall_quality(recall: Any) -> str:
    rate = _as_rate(recall)
    if rate is None:
        return "not available"
    if rate < 0.50:
        return "weak"
    if rate < 0.70:
        return "moderate"
    return "strong"
def _is_concept_query(query: str) -> bool:
    q = _normalize(query)
    if not q:
        return False
    data_markers = (
        "my model", "our model", "this model", "trained model", "current model",
        "show", "results", "metrics", "performance", "prediction", "predicted",
        "actual", "high risk", "how many", "which customers", "is churn high",
        "churn rate", "is my", "is our",
        "current churn", "what is the churn", "churn by", "by segment",
        "model choice", "why this model", "why model",
    )
    if any(marker in q for marker in data_markers):
        return False
    concept_terms = (
        "churn", "recall", "precision", "accuracy", "f1", "f1 score",
        "roc", "auc", "roc-auc", "overfitting", "overfit", "underfitting",
        "underfit", "ml model", "machine learning model", "model",
    )
    if q in concept_terms:
        return True
    concept_markers = (
        "what is", "what does", "define", "definition", "meaning",
        "means", "mean", "explain",
    )
    return any(marker in q for marker in concept_markers) and any(term in q for term in concept_terms)
def _top_feature_names(context: Dict[str, Any], top_n: int = 3) -> List[str]:
    features = _shortlist_top_features_from_shap(context.get("shap"), top_n=top_n)
    names = []
    for f in features:
        if isinstance(f, dict):
            name = str(f.get("name") or "").strip()
        else:
            name = str(f).strip()
        if name:
            names.append(name)
    return names
def _churn_model_context_lines(context: Dict[str, Any], include_missing: bool = False) -> List[str]:
    lines: List[str] = []
    model_type = _get_model_type(context)
    actual = _get_actual_churn_rate(context)
    predicted = _get_predicted_churn_rate(context)
    metrics = context.get("metrics") or {}
    predictions = context.get("predictions") or {}
    features = _top_feature_names(context, top_n=3)
    if model_type or include_missing:
        lines.append(f"- Model: {model_type or 'not trained yet'}")
    if actual is not None or include_missing:
        lines.append(f"- Actual churn rate: {_fmt_pct(actual)}")
    if predicted is not None or include_missing:
        lines.append(f"- Predicted churn rate: {_fmt_pct(predicted)}")
    if isinstance(metrics, dict) and metrics:
        metric_parts = []
        for key, label in (("accuracy", "accuracy"), ("recall", "recall"), ("f1", "F1"), ("roc_auc", "ROC-AUC")):
            val = metrics.get(key)
            if val is None:
                continue
            try:
                if key == "roc_auc":
                    metric_parts.append(f"{label} {float(val):.3f}")
                else:
                    metric_parts.append(f"{label} {_fmt_pct(_as_rate(val))}")
            except Exception:
                metric_parts.append(f"{label} {val}")
        if metric_parts:
            lines.append(f"- Model metrics: {', '.join(metric_parts)}")
    elif include_missing:
        lines.append("- Model metrics: not available yet")
    if isinstance(predictions, dict) and predictions:
        high = predictions.get("high_risk")
        total = predictions.get("total_customers") or predictions.get("total")
        if high is not None:
            try:
                high_text = f"{int(high):,}"
                if total:
                    high_text += f" of {int(total):,} scored customers"
                lines.append(f"- High-risk customers: {high_text}")
            except Exception:
                lines.append(f"- High-risk customers: {high}")
    if features:
        lines.append(f"- Top churn drivers: {', '.join(features)}")
    elif include_missing:
        lines.append("- Top churn drivers: not available yet")
    return lines
def _is_data_intent(intent_key: str) -> bool:
    return intent_key in {
        "churn_rate",
        "top_drivers",
        "metrics",
        "dataset_summary",
        "missing_values",
        "predictions_summary",
        "high_risk",
        "simulation",
        "columns",
        "model_info",
        "model_tracking",
        "training_details",
        "overfitting",
        "full_summary",
        "lifecycle",
        "data_stats",
        "available_models",
    }
def _needs_completeness_guard(query: str, intent_key: str = "") -> bool:
    if _is_concept_query(query):
        return False
    if intent_key:
        return False
    q = _normalize(query)
    if any(k in q for k in (
        "churn", "prediction", "predicted", "high risk", "at risk",
        "metric", "accuracy", "recall", "precision", "f1", "roc", "auc",
        "performance", "how good", "driver", "feature importance",
        "fit status",
    )):
        return True
    if "model" in q and any(k in q for k in ("good", "quality", "result", "perform", "predict", "reliable", "trust", "deploy")):
        return True
    return False
def _metric_summary_for_answer(context: Dict[str, Any]) -> str:
    metrics = context.get("metrics") or {}
    if not isinstance(metrics, dict):
        return ""
    parts = []
    for key, label in (
        ("accuracy", "accuracy"),
        ("recall", "recall"),
        ("precision", "precision"),
        ("f1", "F1"),
        ("roc_auc", "ROC-AUC"),
    ):
        val = metrics.get(key)
        if val is None:
            continue
        try:
            if key == "roc_auc":
                parts.append(f"{label} {float(val):.3f}")
            else:
                parts.append(f"{label} {_fmt_pct(_as_rate(val))}")
        except Exception:
            parts.append(f"{label} {val}")
    return ", ".join(parts)
def _build_grounded_complete_answer(query: str, context: Dict[str, Any], intent_key: str = "") -> str:
    context = context or {}
    q = _normalize(query)
    lines: List[str] = []
    model_type = _get_model_type(context)
    actual = _get_actual_churn_rate(context)
    predicted = _get_predicted_churn_rate(context)
    metrics = context.get("metrics") or {}
    predictions = context.get("predictions") or {}
    features = _top_feature_names(context, top_n=3)
    recall = metrics.get("recall") if isinstance(metrics, dict) else None
    if intent_key == "overfitting" or "overfit" in q or "underfit" in q:
        lines.append("Here is the grounded overfitting read from your model context.")
    elif any(k in q for k in ("reliable", "trust", "deploy")) or (
        "model" in q and any(k in q for k in ("good", "quality", "perform"))
    ):
        recall_rate = _as_rate(recall)
        if recall_rate is None:
            lines.append("I cannot fully judge model reliability yet because recall is not available.")
        elif recall_rate < 0.50:
            lines.append(f"No - this model is not reliable enough for churn decisions yet because recall is only {_fmt_pct(recall_rate)}.")
        elif recall_rate < 0.70:
            lines.append(f"Partly - recall is {_fmt_pct(recall_rate)}, so the model captures most churners but can still improve.")
        else:
            lines.append(f"Yes - recall is {_fmt_pct(recall_rate)}, so the model catches most real churners.")
    elif intent_key in {"churn_rate", "predictions_summary", "high_risk"} or "churn" in q or "predict" in q:
        if actual is not None and predicted is not None:
            diff = (predicted - actual) * 100
            direction = "underestimating" if diff < 0 else "overestimating" if diff > 0 else "matching"
            lines.append(
                f"Actual churn is {_fmt_pct(actual)} and predicted churn is {_fmt_pct(predicted)}. "
                f"The model is {direction} churn by {abs(diff):.1f} percentage points."
            )
        elif actual is not None:
            lines.append(f"Actual churn is {_fmt_pct(actual)}. Predicted churn is not available yet.")
        elif predicted is not None:
            lines.append(f"Predicted churn is {_fmt_pct(predicted)}. Actual churn is not available yet.")
        else:
            lines.append("Churn rate is not available yet from the current context.")
    elif intent_key == "top_drivers" or "driver" in q or "feature" in q:
        if features:
            lines.append("The strongest churn drivers currently identified are " + ", ".join(features) + ".")
        else:
            lines.append("Top churn drivers are not available yet.")
    else:
        lines.append("Here is the grounded answer using the available churn context.")
    facts: List[str] = []
    if model_type:
        facts.append(f"- Model: {model_type}")
    if actual is not None:
        facts.append(f"- Actual churn rate: {_fmt_pct(actual)}")
    if predicted is not None:
        facts.append(f"- Predicted churn rate: {_fmt_pct(predicted)}")
    metric_summary = _metric_summary_for_answer(context)
    if metric_summary:
        facts.append(f"- Model metrics: {metric_summary}")
    if isinstance(predictions, dict):
        high = predictions.get("high_risk")
        total = predictions.get("total_customers") or predictions.get("total")
        if high is not None:
            try:
                high_text = f"{int(high):,}"
                if total:
                    high_text += f" of {int(total):,} scored customers"
                facts.append(f"- High-risk customers: {high_text}")
            except Exception:
                facts.append(f"- High-risk customers: {high}")
    if features:
        facts.append(f"- Top churn drivers: {', '.join(features)}")
    if facts:
        lines.append("\nKey context:")
        lines.extend(facts)
    if recall is not None:
        lines.append(f"\nBusiness verdict: {_recall_verdict(recall)}")
    elif isinstance(metrics, dict) and metrics:
        lines.append("\nBusiness verdict: Recall is not available, so model usefulness for catching churners cannot be fully judged yet.")
    if features:
        lines.append(
            "This means retention decisions should focus on the customers affected by those drivers, "
            "while using recall to judge whether the model is catching enough real churners."
        )
    else:
        lines.append("Next step: generate feature importance so the numbers can be connected to specific churn causes.")
    return "\n".join(lines)
def _enforce_answer_completeness(answer: str, query: str, context: Dict[str, Any], intent_key: str = "") -> str:
    if not answer or not _needs_completeness_guard(query, intent_key):
        return answer
    context = context or {}
    answer_lower = _normalize(answer)
    additions: List[str] = []
    model_type = _get_model_type(context)
    if model_type and model_type.lower() not in answer_lower:
        additions.append(f"- Model: {model_type}")
    actual = _get_actual_churn_rate(context)
    if actual is not None and "actual churn" not in answer_lower:
        additions.append(f"- Actual churn rate: {_fmt_pct(actual)}")
    predicted = _get_predicted_churn_rate(context)
    if predicted is not None and "predicted churn" not in answer_lower:
        additions.append(f"- Predicted churn rate: {_fmt_pct(predicted)}")
    metrics = context.get("metrics") or {}
    if isinstance(metrics, dict):
        metric_keys = ("accuracy", "recall", "f1", "roc_auc")
        available_metrics = [k for k in metric_keys if metrics.get(k) is not None]
        missing_metric_labels = [
            k for k in available_metrics
            if (("roc" not in answer_lower and "auc" not in answer_lower) if k == "roc_auc" else k not in answer_lower)
        ]
        metric_summary = _metric_summary_for_answer(context)
        if missing_metric_labels and metric_summary:
            additions.append(f"- Model metrics: {metric_summary}")
        recall = metrics.get("recall")
        if recall is not None and not any(
            phrase in answer_lower
            for phrase in ("business verdict", "recall verdict", "not suitable", "needs improvement", "strong recall", "acceptable")
        ):
            additions.append(f"- Recall verdict: {_recall_verdict(recall)}")
    predictions = context.get("predictions") or {}
    if isinstance(predictions, dict) and intent_key in {"predictions_summary", "high_risk", "full_summary"}:
        high = predictions.get("high_risk")
        total = predictions.get("total_customers") or predictions.get("total")
        if high is not None and "high risk" not in answer_lower:
            try:
                high_text = f"{int(high):,}"
                if total:
                    high_text += f" of {int(total):,} scored customers"
                additions.append(f"- High-risk customers: {high_text}")
            except Exception:
                additions.append(f"- High-risk customers: {high}")
    features = _top_feature_names(context, top_n=3)
    if features and not all(name.lower() in answer_lower for name in features):
        additions.append(f"- Top churn drivers: {', '.join(features)}")
    if not additions:
        return answer
    return _build_grounded_complete_answer(query, context, intent_key)
def _handle_concept_question(query: str, context: Dict[str, Any]) -> str:
    q = _normalize(query)
    metrics = context.get("metrics") or {}
    if "recall" in q:
        lines = [
            "Recall is the percentage of actual positive cases that the model correctly identifies.",
            "",
            "In churn prediction, it means: out of all customers who actually churned, how many did the model catch?",
        ]
        recall = metrics.get("recall") if isinstance(metrics, dict) else None
        recall_rate = _as_rate(recall)
        if recall_rate is not None:
            lines.append(
                f"\nFor your model, recall is {_fmt_pct(recall_rate)}, so it catches about {recall_rate * 100:.0f} out of every 100 churners. "
                f"That is {_recall_quality(recall)} recall."
            )
        return "\n".join(lines)
    if "precision" in q:
        lines = [
            "Precision measures how many customers flagged as churn risks actually churn.",
            "",
            "In churn prediction, high precision means the retention list has fewer false alarms.",
        ]
        precision = metrics.get("precision") if isinstance(metrics, dict) else None
        precision_rate = _as_rate(precision)
        if precision_rate is not None:
            lines.append(f"\nFor your model, precision is {_fmt_pct(precision_rate)}.")
        return "\n".join(lines)
    if "accuracy" in q:
        lines = [
            "Accuracy is the percentage of total predictions the model gets right.",
            "",
            "For churn, accuracy alone can be misleading if churners are a minority; recall is usually more important for catching customers before they leave.",
        ]
        accuracy = metrics.get("accuracy") if isinstance(metrics, dict) else None
        accuracy_rate = _as_rate(accuracy)
        if accuracy_rate is not None:
            lines.append(f"\nFor your model, accuracy is {_fmt_pct(accuracy_rate)}.")
        return "\n".join(lines)
    if "f1" in q:
        lines = [
            "F1 score balances precision and recall into one number.",
            "",
            "In churn prediction, it helps judge whether the model catches churners without creating too many false alarms.",
        ]
        f1 = metrics.get("f1") if isinstance(metrics, dict) else None
        f1_rate = _as_rate(f1)
        if f1_rate is not None:
            lines.append(f"\nFor your model, F1 is {_fmt_pct(f1_rate)}.")
        return "\n".join(lines)
    if "roc" in q or "auc" in q:
        lines = [
            "ROC-AUC measures how well the model ranks risky customers above safer customers.",
            "",
            "A value near 1.0 is excellent; 0.5 means the ranking is no better than random.",
        ]
        roc = _as_float(metrics.get("roc_auc")) if isinstance(metrics, dict) else None
        if roc is not None:
            lines.append(f"\nFor your model, ROC-AUC is {roc:.3f}.")
        return "\n".join(lines)
    if "overfit" in q or "overfitting" in q:
        lines = [
            "Overfitting means a model learns noise or overly specific training patterns instead of general patterns.",
            "",
            "The result is a model that looks good on training data but performs worse on new customers.",
        ]
        train_score = _as_float(metrics.get("train_score") or metrics.get("train_f1")) if isinstance(metrics, dict) else None
        test_score = _as_float(metrics.get("test_score") or metrics.get("f1")) if isinstance(metrics, dict) else None
        if train_score is not None and test_score is not None:
            gap = (train_score - test_score) * 100
            lines.append(
                f"\nIn your case, train score is {_fmt_pct(_as_rate(train_score))} and test score is {_fmt_pct(_as_rate(test_score))}, "
                f"a gap of {gap:.1f} percentage points."
            )
        return "\n".join(lines)
    if "underfit" in q or "underfitting" in q:
        lines = [
            "Underfitting means a model is too simple to learn the real patterns in the data.",
            "",
            "It usually performs poorly on both training data and new data.",
        ]
        fit_status = metrics.get("fit_status") if isinstance(metrics, dict) else None
        if fit_status:
            lines.append(f"\nYour current fit status is {fit_status}.")
        return "\n".join(lines)
    if "churn" in q:
        lines = [
            "Churn means customers leaving your service, cancelling, or stopping usage.",
            "",
            "In this system, churn is the outcome the model tries to predict so the business can intervene earlier.",
        ]
        actual = _get_actual_churn_rate(context)
        if actual is not None:
            lines.append(f"\nIn your dataset, actual churn is {_fmt_pct(actual)}.")
        return "\n".join(lines)
    if "model" in q:
        model_type = _get_model_type(context)
        lines = [
            "A machine learning model is a trained pattern-recognition system that learns from historical data and makes predictions on new records.",
            "",
            "Here, the model learns customer patterns linked to churn and outputs churn risk.",
        ]
        if model_type:
            lines.append(f"\nYour current model is {model_type}.")
        return "\n".join(lines)
    return "This is a general ML concept question. Ask about a specific term like recall, churn, overfitting, precision, or ROC-AUC and I will explain it in churn context."
# -------------------------
# PII redaction
# -------------------------
PII_PATTERNS = [
    re.compile(r"[\w\.-]+@[\w\.-]+\.\w+"),       # email
    re.compile(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"),  # ssn-like
    re.compile(r"\b\d{10}\b"),                   # phone-like 10 digits
]
def redact_text(text: str) -> str:
    if not text:
        return text
    out = text
    for p in PII_PATTERNS:
        out = p.sub("[REDACTED]", out)
    return out
def redact_context(context: Dict[str, Any]) -> Dict[str, Any]:
    """Return a small redacted summary for sending to an external LLM."""
    safe = {}
    if not context:
        return safe
    eda = context.get("eda")
    if eda and isinstance(eda, dict):
        safe_eda = {}
        if "shape" in eda:
            safe_eda["shape"] = eda.get("shape")
        if "n_rows" in eda:
            safe_eda["n_rows"] = eda.get("n_rows")
        if "n_cols" in eda:
            safe_eda["n_cols"] = eda.get("n_cols")
        cols = eda.get("columns")
        if cols:
            names = []
            if isinstance(cols, list):
                for c in cols:
                    if isinstance(c, dict) and "name" in c:
                        names.append(str(c["name"]))
                    elif isinstance(c, str):
                        names.append(c)
            safe_eda["columns"] = names[:200]
        if "missing_total" in eda:
            safe_eda["missing_total"] = eda.get("missing_total")
        if "target_column" in eda:
            safe_eda["target_column"] = eda.get("target_column")
        if "churn_rate" in eda:
            safe_eda["churn_rate"] = eda.get("churn_rate")
        safe["eda"] = safe_eda
    metrics = context.get("metrics")
    if metrics and isinstance(metrics, dict):
        safe_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                safe_metrics[k] = float(v)
            elif isinstance(v, str) and len(v) < 200:
                safe_metrics[k] = redact_text(v)
        safe["metrics"] = safe_metrics
    shap = context.get("shap")
    if shap and isinstance(shap, dict):
        top = shap.get("top_features") or shap.get("features") or []
        safe_top = []
        for f in top[:50]:
            if isinstance(f, dict):
                name = str(f.get("name", ""))
                mean_abs = f.get("mean_abs_shap")
                try:
                    safe_top.append({"name": name, "mean_abs_shap": float(mean_abs) if mean_abs is not None else None})
                except Exception:
                    safe_top.append({"name": name, "mean_abs_shap": None})
            else:
                safe_top.append({"name": str(f)})
        safe["shap"] = {"top_features": safe_top}
    sim = context.get("simulate")
    if sim and isinstance(sim, dict):
        safe_sim = {}
        for k in ("before_churn_rate", "after_churn_rate", "retained_customers", "revenue_saved", "action_cost", "roi"):
            if k in sim and isinstance(sim[k], (int, float)):
                safe_sim[k] = float(sim[k])
        safe["simulate"] = safe_sim
    # Include current_model explicitly
    current_model = context.get("current_model") or (metrics and metrics.get("model_type")) or ""
    if current_model:
        safe["current_model"] = str(current_model)
    # Include available_models (what the system supports, not what has been trained)
    available = context.get("available_models")
    if available and isinstance(available, list):
        safe["available_models"] = [str(m) for m in available]
    else:
        # Hard-coded fallback so the LLM always knows supported algorithms
        safe["available_models"] = [
            "Logistic Regression",
            "Random Forest",
            "XGBoost",
            "Decision Tree",
            "Gradient Boosting",
            "AdaBoost",
            "Naive Bayes",
            "CatBoost",
        ]
    # Include model history summary
    history = context.get("model_history")
    if history and isinstance(history, list):
        safe_history = []
        for h in history[-5:]:
            if isinstance(h, dict):
                entry = {
                    "model_type": h.get("model_type") or h.get("model", ""),
                    "suspicious": bool(h.get("suspicious")),
                }
                m2 = h.get("metrics") or h
                if isinstance(m2, dict):
                    entry["metrics"] = {
                        k: float(v) for k, v in m2.items()
                        if isinstance(v, (int, float))
                    }
                safe_history.append(entry)
        safe["model_history"] = safe_history
    # Include predictions summary
    predictions = context.get("predictions")
    if predictions and isinstance(predictions, dict):
        safe_pred = {}
        for k in ("total_customers", "total", "high_risk", "medium_risk", "low_risk",
                  "predicted_churn_rate", "predicted_churn", "revenue_at_risk",
                  "churn_count", "churn_rate"):
            if k in predictions:
                try:
                    safe_pred[k] = float(predictions[k])
                except Exception:
                    pass
        # Also include probability_summary if present
        prob_summary = predictions.get("probability_summary")
        if isinstance(prob_summary, dict):
            safe_pred["probability_summary"] = {
                k: float(v) for k, v in prob_summary.items()
                if isinstance(v, (int, float))
            }
        safe["predictions"] = safe_pred
    # Include lifecycle summary
    lifecycle = context.get("lifecycle")
    if lifecycle and isinstance(lifecycle, dict):
        safe["lifecycle"] = {
            "column_used": lifecycle.get("column_used", ""),
            "highest_risk_segment": lifecycle.get("highest_risk_segment", ""),
        }
        # Also include per-stage stats if available (no raw data, just aggregated rates)
        raw_segments = lifecycle.get("segments")
        if isinstance(raw_segments, dict):
            safe_stages = {}
            for stage, stats in raw_segments.items():
                if isinstance(stats, dict):
                    safe_stages[stage] = {
                        k: float(v) for k, v in stats.items()
                        if isinstance(v, (int, float)) and k in ("customers", "churned", "not_churned", "churn_rate")
                    }
            if safe_stages:
                safe["lifecycle"]["segments"] = safe_stages
    # Include segment analysis result
    segment_result = (
        context.get("segment_result")
        or context.get("segment")
        or context.get("time_churn")
    )
    if segment_result and isinstance(segment_result, dict):
        safe_seg: Dict[str, Any] = {}
        safe_seg["column"] = str(segment_result.get("column") or
                                  segment_result.get("result", {}).get("column") or "")
        safe_seg["mode"] = str(segment_result.get("mode") or "")
        raw_segs = (
            segment_result.get("segments")
            or segment_result.get("result", {}).get("segments")
            or []
        )
        safe_segs = []
        if isinstance(raw_segs, list):
            for s in raw_segs[:20]:
                if isinstance(s, dict):
                    entry: Dict[str, Any] = {"value": str(s.get("value", ""))}
                    for fld in ("customers", "churned", "not_churned", "churn_rate"):
                        if fld in s:
                            try:
                                entry[fld] = float(s[fld])
                            except Exception:
                                pass
                    safe_segs.append(entry)
        safe_seg["segments"] = safe_segs
        if safe_seg["column"] or safe_segs:
            safe["segment"] = safe_seg
    return safe
# -------------------------
# Model explanation dictionary
# -------------------------
MODEL_EXPLANATIONS = {
    "Logistic Regression": (
        "Simple linear model — fast, interpretable, and robust. "
        "It works well for churn because churn is often driven by a combination of additive factors "
        "(payment delay + support calls + low tenure). Its low train/test gap makes it trustworthy for deployment."
    ),
    "Random Forest": (
        "Tree-based ensemble model that handles non-linear patterns and reduces overfitting by averaging many trees. "
        "Good at capturing complex interactions between features like tenure, usage frequency, and payment history."
    ),
    "XGBoost": (
        "Advanced gradient boosting model — high performance on tabular data. "
        "Sequentially corrects prediction errors, making it very effective for churn with complex patterns, "
        "but can overfit on small or simple datasets."
    ),
    "Decision Tree": (
        "Simple, interpretable model that splits data on feature thresholds. "
        "Easy to explain to stakeholders, but prone to overfitting on training data."
    ),
    "Naive Bayes": (
        "Fast probabilistic model that assumes features are independent. "
        "Works surprisingly well for churn when features have weak correlations, "
        "and is highly resistant to overfitting."
    ),
    "Gradient Boosting": (
        "Sequential boosting model that builds trees one at a time, correcting previous errors. "
        "Strong performance but slower to train than XGBoost."
    ),
    "AdaBoost": (
        "Boosting model that focuses on correcting misclassified customers. "
        "Achieves very high recall (catches most churners) but at the cost of lower precision."
    ),
    "CatBoost": (
        "Advanced boosting model with built-in handling for categorical features. "
        "Very powerful but frequently achieves suspiciously perfect metrics on churn datasets, "
        "suggesting possible data leakage or trivially separable class boundaries."
    ),
}
def _handle_help(_context: Dict[str, Any]) -> str:
    return (
        "Here's what I can help you with:\n\n"
        "- **Dataset overview** — 'What does my dataset contain?' / 'How many rows?'\n"
        "- **Missing values** — 'Are there missing values in my data?'\n"
        "- **Churn rate** — 'What is the current churn rate?'\n"
        "- **Why customers are leaving** — 'Why are customers churning?'\n"
        "- **Model performance** — 'How good is the model?' / 'Which model did I train?'\n"
        "- **Predictions** — 'How many customers are predicted to churn?' / 'Show high risk customers'\n"
        "- **Segment analysis** — 'What is churn by segment?' / 'Analyze churn by category'\n"
        "- **Lifecycle risk** — 'What is the lifecycle risk?'\n"
        "- **Retention simulation** — 'What was the ROI of the last simulation?'\n"
        "- **What to do next** — 'What actions should we take?'\n"
        "- **Full summary** — 'Show me all results'\n\n"
        "For individual customer explanations, use the **Explain** tab and enter a row number."
    )
def _handle_churn_rate(context: Dict[str, Any]) -> str:
    actual_rate = _get_actual_churn_rate(context)
    predicted_rate = _get_predicted_churn_rate(context)
    recall = _get_metric(context, "recall")
    features = _top_feature_names(context, top_n=3)
    if actual_rate is None and predicted_rate is None:
        return (
            "Churn rate is not available yet.\n\n"
            "Actual churn rate becomes available after training because it comes from the target labels. "
            "Predicted churn rate becomes available after running predictions."
        )
    lines: List[str] = []
    if actual_rate is not None:
        level = "high" if actual_rate > 0.4 else "moderate" if actual_rate > 0.2 else "relatively low"
        lines.append(f"**Actual churn rate: {_fmt_pct(actual_rate, 2)}**")
        lines.append(f"This is the ground-truth churn rate from your training labels, and it is {level}.")
    else:
        lines.append("**Actual churn rate: not available yet**")
        lines.append("Train a model first; the actual churn rate comes from the selected target column.")
    if predicted_rate is not None:
        lines.append(f"**Predicted churn rate: {_fmt_pct(predicted_rate, 2)}**")
    else:
        lines.append("**Predicted churn rate: not available yet**")
        lines.append("Run predictions after training to compare model output against actual churn.")
    if actual_rate is not None and predicted_rate is not None:
        diff = (predicted_rate - actual_rate) * 100
        direction = "underestimating" if diff < 0 else "overestimating" if diff > 0 else "matching"
        lines.append(
            f"\nThe model is {direction} churn by {abs(diff):.1f} percentage points. "
            "This matters because underestimation means real at-risk customers may be missed, while overestimation can create extra false alarms."
        )
    if recall is not None:
        lines.append(f"\nRecall context: {_fmt_pct(_as_rate(recall))}. {_recall_verdict(recall)}")
    if features:
        lines.append(
            "\nTop churn drivers currently identified: "
            + ", ".join(features)
            + ". These are the factors to investigate when explaining why churn is happening."
        )
    lines.append("\nNext step: use recall and the top drivers together before acting on the predicted churn list.")
    return "\n".join(lines)
def _handle_top_drivers(context: Dict[str, Any]) -> str:
    features = _shortlist_top_features_from_shap(context.get("shap"), top_n=5)
    if not features:
        return (
            "Top churn drivers are not available yet.\n\n"
            "Train a model first so the system can compute feature importance and explain why customers are likely to churn."
        )
    lines = ["Here is what is driving churn in your current model:\n"]
    for i, f in enumerate(features):
        name = f.get("name", "<unknown>") if isinstance(f, dict) else str(f)
        score = _as_float(f.get("mean_abs_shap")) if isinstance(f, dict) else None
        if score is not None:
            lines.append(f"{i + 1}. **{name}** - influence score {score:.4f}")
        else:
            lines.append(f"{i + 1}. **{name}**")
    context_lines = _churn_model_context_lines(context, include_missing=False)
    if context_lines:
        lines.append("\nRelevant model context:")
        lines.extend(context_lines)
    lines.append(
        "\nBusiness meaning: customers with risky patterns in these drivers should be prioritized for targeted retention actions, especially if recall shows the model is missing churners."
    )
    return "\n".join(lines)
def _handle_metrics(context: Dict[str, Any]) -> str:
    m = context.get("metrics") or {}
    if not m:
        return (
            "Model metrics are not available yet because no model has been trained in this session.\n\n"
            "Train a model first. After training, I can explain accuracy, recall, precision, F1, ROC-AUC, actual churn rate, and how the result connects to your churn risk."
        )
    lines = ["Here is what the current model performance means:\n"]
    model_type = _get_model_type(context)
    if model_type:
        lines.append(f"- **Model:** {model_type}")
    actual = _get_actual_churn_rate(context)
    predicted = _get_predicted_churn_rate(context)
    if actual is not None:
        lines.append(f"- **Actual churn rate:** {_fmt_pct(actual)}")
    if predicted is not None:
        lines.append(f"- **Predicted churn rate:** {_fmt_pct(predicted)}")
    if actual is not None and predicted is not None:
        diff = (predicted - actual) * 100
        direction = "underestimates" if diff < 0 else "overestimates" if diff > 0 else "matches"
        lines.append(f"- **Actual vs predicted:** the model {direction} churn by {abs(diff):.1f} percentage points")
    metric_specs = (
        ("accuracy", "Accuracy", "overall correctness"),
        ("recall", "Recall", "how many true churners the model catches"),
        ("precision", "Precision", "how clean the at-risk list is"),
        ("f1", "F1 score", "balance between recall and precision"),
    )
    for key, label, meaning in metric_specs:
        val = m.get(key)
        if val is None:
            continue
        rate = _as_rate(val)
        if rate is not None:
            lines.append(f"- **{label}: {_fmt_pct(rate)}** - {meaning}")
    roc = _as_float(m.get("roc_auc"))
    if roc is not None:
        quality = "excellent" if roc >= 0.9 else "good" if roc >= 0.8 else "fair" if roc >= 0.7 else "weak"
        lines.append(f"- **ROC-AUC: {roc:.3f}** - {quality} ability to rank risky customers above safer customers")
    fit_status = m.get("fit_status") or m.get("fit")
    if fit_status:
        lines.append(f"- **Fit status:** {fit_status}")
    lines.append(f"\n**Business verdict:** {_recall_verdict(m.get('recall'))}")
    features = _top_feature_names(context, top_n=3)
    if features:
        lines.append(
            "The main churn drivers behind the model are "
            + ", ".join(features)
            + ", so performance should be interpreted around whether the model is catching customers affected by those factors."
        )
    else:
        lines.append("Top churn drivers are not available yet; train/refresh model explanation to connect metrics to causes.")
    lines.append("Next step: optimize for recall if the business goal is to catch more at-risk customers before they leave.")
    return "\n".join(lines)
def _handle_simulation(context: Dict[str, Any]) -> str:
    sim = context.get("simulate")
    if not sim:
        return (
            "No retention simulation has been run yet.\n\n"
            "Go to the **Simulate** tab and enter a discount percentage, contract extension length, "
            "and cost per customer. I'll summarize the projected impact once it's done."
        )
    return "Here's a summary of the last retention simulation:\n\n" + _format_simulation(sim)
def _handle_columns(context: Dict[str, Any]) -> str:
    eda = context.get("eda") or {}
    cols = eda.get("columns")
    if not cols:
        return (
            "No dataset has been uploaded yet.\n\n"
            "Upload a CSV or Excel file in the **Upload** tab — I'll summarize the columns and data structure right away."
        )
    if isinstance(cols, list):
        names = [c["name"] if isinstance(c, dict) and "name" in c else str(c) for c in cols]
        col_list = ", ".join(names[:50]) + ("..." if len(names) > 50 else "")
        target = eda.get("target_column", "")
        target_note = f"\n\n**Target column:** {target}" if target else ""
        return (
            f"Your dataset has **{len(names)} columns**:\n\n"
            f"{col_list}"
            f"{target_note}\n\n"
            f"Use the **EDA** tab for a full breakdown of data types, missing values, and distributions."
        )
    return "Column list is present but could not be formatted. Check the EDA tab for the full view."
def _handle_explain_customer(_context: Dict[str, Any]) -> str:
    return (
        "For individual customer explanations, please use the **Explain** tab.\n\n"
        "Enter a row number from your dataset, and the system will show you exactly which factors "
        "pushed that customer's churn risk up or down — without exposing any personal data here in chat."
    )
def _handle_recommend_actions(context: Dict[str, Any]) -> str:
    shap = context.get("shap") or {}
    top = (shap.get("top_features") if isinstance(shap, dict) else None) or []
    sim = context.get("simulate")
    lines = []
    if top:
        lines.append("Based on the analysis, here are the most impactful actions you can take:\n")
        for f in top[:3]:
            name = f.get("name") if isinstance(f, dict) else str(f)
            action = (
                f"Customers influenced by **{name}** show higher churn risk. "
                "Analyze this group closely and take targeted action early to reduce potential churn."
            )
            lines.append(f"- **{name}**: {action}")
    else:
        lines.append(
            "To get specific, data-driven recommendations, train a model first.\n\n"
            "Once training is complete, the system will identify which factors most influence churn in your dataset "
            "and I'll turn those into clear actions for your team."
        )
    if sim:
        lines.append("\n**Last simulation results:**")
        try:
            if "roi" in sim:
                roi_val = float(sim['roi'])
                lines.append(f"- Projected ROI: {roi_val:.2f}% {'✅ positive return' if roi_val > 0 else '⚠️ review campaign settings'}")
        except Exception:
            pass
        try:
            if "retained_customers" in sim:
                lines.append(f"- Customers projected to be retained: {int(sim['retained_customers']):,}")
        except Exception:
            pass
    lines.append("\nRun a simulation in the **Simulate** tab to estimate the financial return of any of these actions before committing budget.")
    return "\n".join(lines)
# -------------------------
# NEW HANDLERS
# -------------------------
def _handle_dataset_summary(context: Dict[str, Any]) -> str:
    """Answer questions about the uploaded dataset — rows, columns, types, target."""
    eda = context.get("eda") or {}
    if not eda:
        return (
            "No dataset has been loaded yet.\n\n"
            "Please upload a CSV or Excel file in the **Upload** tab to get started."
        )
    n_rows = eda.get("n_rows") or (eda.get("shape") or [None])[0]
    n_cols = eda.get("n_cols") or (eda.get("shape") or [None, None])[1]
    cols = eda.get("columns") or []
    missing_total = eda.get("missing_total", 0)
    target = eda.get("target_column", "")
    col_names = []
    if isinstance(cols, list):
        for c in cols:
            if isinstance(c, dict) and "name" in c:
                col_names.append(str(c["name"]))
            elif isinstance(c, str):
                col_names.append(c)
    lines = ["Here's a quick overview of your uploaded dataset:\n"]
    if n_rows:
        try:
            lines.append(f"- **Rows:** {int(n_rows):,} — total number of customer records")
        except Exception:
            lines.append(f"- **Rows:** {n_rows}")
    if n_cols or col_names:
        count = len(col_names) if col_names else n_cols
        lines.append(f"- **Columns:** {count} — features available for analysis")
    if col_names:
        sample = ", ".join(col_names[:15])
        suffix = f" ... and {len(col_names) - 15} more" if len(col_names) > 15 else ""
        lines.append(f"- **Column names:** {sample}{suffix}")
    if target:
        lines.append(f"- **Target column:** {target} — this is the churn label the model learns to predict")
    if missing_total is not None:
        try:
            mv = int(missing_total)
            if mv == 0:
                lines.append("- **Missing values:** None — your dataset is complete")
            else:
                lines.append(f"- **Missing values:** {mv:,} total across all columns — the system handles these automatically during preprocessing")
        except Exception:
            pass
    churn_rate = eda.get("churn_rate") or eda.get("target_churn_rate")
    if churn_rate is not None:
        try:
            rate = float(churn_rate)
            pct = rate * 100 if rate <= 1 else rate
            lines.append(f"- **Observed churn rate:** {pct:.1f}%")
        except Exception:
            pass
    if len(lines) <= 1:
        return (
            "A dataset is loaded but detailed metadata isn't available yet.\n\n"
            "Run **Quick EDA** or **Full EDA** for a complete breakdown of your data."
        )
    lines.append("\nRun **Full EDA** to see distributions, correlation matrix, and category breakdowns.")
    return "\n".join(lines)
def _handle_missing_values(context: Dict[str, Any]) -> str:
    """Answer questions about missing values in the dataset."""
    eda = context.get("eda") or {}
    if not eda:
        return (
            "No dataset has been uploaded yet.\n\n"
            "Upload a file in the **Upload** tab and the system will automatically detect missing values."
        )
    missing_total = eda.get("missing_total")
    cols = eda.get("columns") or []
    missing_cols = []
    if isinstance(cols, list):
        for c in cols:
            if isinstance(c, dict):
                mv = c.get("missing") or c.get("missing_count") or c.get("null_count")
                name = c.get("name", "")
                if mv and int(mv) > 0:
                    missing_cols.append((name, int(mv)))
    if missing_total is not None:
        try:
            mv = int(missing_total)
            if mv == 0:
                return (
                    "✅ **Your dataset has no missing values.**\n\n"
                    "All columns are complete — no imputation is needed. The model can use all features directly."
                )
            else:
                lines = [f"Your dataset has **{mv:,} missing values** in total.\n"]
                if missing_cols:
                    lines.append("Columns with missing data:")
                    for col_name, cnt in missing_cols[:10]:
                        lines.append(f"- **{col_name}**: {cnt:,} missing")
                    if len(missing_cols) > 10:
                        lines.append(f"  ... and {len(missing_cols) - 10} more columns")
                lines.append(
                    "\nThe preprocessing pipeline automatically fills missing numeric values with the column median "
                    "and handles unseen categories — no manual action is needed."
                )
                return "\n".join(lines)
        except Exception:
            pass
    # If no missing info in EDA, give generic guidance
    return (
        "Missing value details aren't available yet.\n\n"
        "Run **Full EDA** from the EDA section — it will show per-column missing counts. "
        "The system automatically handles missing data during preprocessing."
    )
def _handle_model_info(context: Dict[str, Any]) -> str:
    """Answer questions about which models were trained and how many."""
    history = context.get("model_history")
    metrics = context.get("metrics") or {}
    model_type = metrics.get("model_type") or context.get("model_type") or ""
    if not history and not model_type:
        return (
            "No model has been trained yet in this session.\n\n"
            "Go to the **Train** tab, select a target column and an algorithm, then click Train. "
            "You can train multiple models and compare them automatically."
        )
    lines = []
    if history and isinstance(history, list):
        total = len(history)
        suspicious = [h for h in history if h.get("suspicious")]
        valid = [h for h in history if not h.get("suspicious")]
        lines.append(f"You have trained **{total} model(s)** in this session.\n")
        if suspicious:
            lines.append(
                f"- **{len(suspicious)} model(s) were flagged as suspicious** (near-perfect metrics like 100% accuracy) "
                "and excluded from automatic selection — this typically indicates data leakage or overly simple class separation."
            )
        if valid:
            lines.append(f"- **{len(valid)} valid model(s)** available for use:\n")
            for i, h in enumerate(valid[-5:], 1):
                name = h.get("model_type") or h.get("model") or f"Model {i}"
                m2 = h.get("metrics") or {}
                roc = m2.get("roc_auc")
                recall = m2.get("recall")
                mode = h.get("mode") or h.get("training_mode") or ""
                fit = m2.get("fit_status") or h.get("fit_status") or ""
                detail_parts = []
                if roc is not None:
                    try:
                        detail_parts.append(f"ROC-AUC: {float(roc):.3f}")
                    except Exception:
                        pass
                if recall is not None:
                    try:
                        r_pct = float(recall) * 100 if float(recall) <= 1 else float(recall)
                        detail_parts.append(f"Recall: {r_pct:.1f}%")
                    except Exception:
                        pass
                if mode:
                    detail_parts.append(f"Mode: {mode}")
                if fit:
                    detail_parts.append(f"Fit: {fit}")
                detail = " | ".join(detail_parts)
                lines.append(f"  {i}. **{name}**" + (f" — {detail}" if detail else ""))
    elif model_type:
        lines.append(f"The currently active model is **{model_type}**.\n")
        if metrics:
            lines.append(_format_metrics(metrics))
    lines.append(
        "\nYou can train more models by going back to the **Train** tab and selecting a different algorithm. "
        "The system compares all trained models and recommends the best one automatically."
    )
    return "\n".join(lines)
def _handle_segment_analysis(_context: Dict[str, Any]) -> str:
    """Segment analysis is available in LLM mode only."""
    return (
        "Segment analysis insights are available in **AI assistant mode** only.\n\n"
        "Enable the AI assistant toggle (🤖) at the top of the chat panel, then ask again — "
        "the AI will explain churn breakdowns by any column in your dataset with full context."
    )
def _handle_predictions_summary(context: Dict[str, Any]) -> str:
    predictions = context.get("predictions") or {}
    if not predictions:
        return (
            "Predictions are not available yet.\n\n"
            "Train a model, then run predictions. After that I can report predicted churn rate, churn count, risk tiers, and how those outputs compare with actual churn."
        )
    lines = ["Here are the prediction results for your dataset:\n"]
    total = predictions.get("total_customers") or predictions.get("total")
    churn_count = predictions.get("churn_count")
    high = predictions.get("high_risk")
    medium = predictions.get("medium_risk")
    low = predictions.get("low_risk")
    predicted_rate = _get_predicted_churn_rate(context)
    actual_rate = _get_actual_churn_rate(context)
    revenue_at_risk = predictions.get("revenue_at_risk")
    if total is not None:
        try:
            lines.append(f"- **Total customers scored:** {int(total):,}")
        except Exception:
            lines.append(f"- **Total customers scored:** {total}")
    if churn_count is not None:
        try:
            lines.append(f"- **Predicted churners:** {int(churn_count):,}")
        except Exception:
            lines.append(f"- **Predicted churners:** {churn_count}")
    if predicted_rate is not None:
        level = "high" if predicted_rate > 0.4 else "moderate" if predicted_rate > 0.2 else "low"
        lines.append(f"- **Predicted churn rate:** {_fmt_pct(predicted_rate)} - this is {level}")
    else:
        lines.append("- **Predicted churn rate:** not available in the prediction summary")
    if actual_rate is not None:
        lines.append(f"- **Actual churn rate:** {_fmt_pct(actual_rate)}")
    else:
        lines.append("- **Actual churn rate:** not available yet because training metrics are missing")
    if actual_rate is not None and predicted_rate is not None:
        diff = (predicted_rate - actual_rate) * 100
        direction = "underestimating" if diff < 0 else "overestimating" if diff > 0 else "matching"
        lines.append(f"- **Comparison:** the model is {direction} churn by {abs(diff):.1f} percentage points")
    for label, value, meaning in (
        ("High risk", high, "prioritize these first"),
        ("Medium risk", medium, "monitor or target selectively"),
        ("Low risk", low, "low immediate action needed"),
    ):
        if value is not None:
            try:
                lines.append(f"- **{label}:** {int(value):,} customers - {meaning}")
            except Exception:
                lines.append(f"- **{label}:** {value} - {meaning}")
    if revenue_at_risk is not None:
        try:
            lines.append(f"- **Revenue at risk:** ${float(revenue_at_risk):,.2f}")
        except Exception:
            lines.append(f"- **Revenue at risk:** {revenue_at_risk}")
    recall = _get_metric(context, "recall")
    if recall is not None:
        lines.append(f"\nRecall check: {_fmt_pct(_as_rate(recall))}. {_recall_verdict(recall)}")
    features = _top_feature_names(context, top_n=3)
    if features:
        lines.append("Top drivers to inspect for these risky customers: " + ", ".join(features) + ".")
    lines.append("Next step: use the Simulate tab to estimate how much of this predicted churn can be prevented.")
    return "\n".join(lines)
    """Answer questions about prediction results — how many customers, risk tiers, etc."""
    predictions = context.get("predictions") or {}
    if not predictions:
        return (
            "No predictions have been run yet.\n\n"
            "After training a model, go to the **Predict** tab and click **Run Prediction**. "
            "The system will score every customer with a churn probability and assign risk tiers (High / Medium / Low)."
        )
    lines = ["Here are the prediction results for your dataset:\n"]
    total = predictions.get("total_customers") or predictions.get("total")
    high = predictions.get("high_risk")
    medium = predictions.get("medium_risk")
    low = predictions.get("low_risk")
    churn_rate = predictions.get("predicted_churn_rate") or predictions.get("predicted_churn")
    revenue_at_risk = predictions.get("revenue_at_risk")
    if total:
        try:
            lines.append(f"- **Total customers scored:** {int(total):,}")
        except Exception:
            lines.append(f"- Total customers scored: {total}")
    if churn_rate is not None:
        try:
            rate = float(churn_rate)
            pct = rate * 100 if rate <= 1 else rate
            level = "high" if pct > 40 else "moderate" if pct > 20 else "low"
            lines.append(f"- **Predicted churn rate:** {pct:.1f}% — this is {level}")
        except Exception:
            lines.append(f"- Predicted churn rate: {churn_rate}")
    if high is not None:
        try:
            lines.append(
                f"- 🔴 **High risk:** {int(high):,} customers (probability ≥ 0.70) — "
                "these should be prioritized immediately for retention intervention"
            )
        except Exception:
            lines.append(f"- High risk: {high}")
    if medium is not None:
        try:
            lines.append(
                f"- 🟡 **Medium risk:** {int(medium):,} customers (probability 0.35–0.70) — "
                "monitor these and consider selective outreach"
            )
        except Exception:
            lines.append(f"- Medium risk: {medium}")
    if low is not None:
        try:
            lines.append(
                f"- 🟢 **Low risk:** {int(low):,} customers (probability < 0.35) — "
                "these are stable and require minimal immediate action"
            )
        except Exception:
            lines.append(f"- Low risk: {low}")
    if revenue_at_risk is not None:
        try:
            lines.append(f"- 💸 **Revenue at risk:** ${float(revenue_at_risk):,.2f} — estimated from predicted churners")
        except Exception:
            lines.append(f"- Revenue at risk: {revenue_at_risk}")
    lines.append(
        "\nUse the **Simulate** tab to estimate how much of this churn can be prevented and at what ROI."
    )
    return "\n".join(lines)
def _handle_high_risk(context: Dict[str, Any]) -> str:
    """Answer questions specifically about high-risk customers."""
    predictions = context.get("predictions") or {}
    if not predictions:
        return (
            "No predictions have been run yet.\n\n"
            "Train a model and run predictions — the system will classify every customer as "
            "High Risk (≥70% churn probability), Medium Risk, or Low Risk."
        )
    high = predictions.get("high_risk")
    total = predictions.get("total_customers") or predictions.get("total")
    revenue_at_risk = predictions.get("revenue_at_risk")
    lines = ["**High-Risk Customer Summary:**\n"]
    if high is not None:
        try:
            h = int(high)
            lines.append(f"- **{h:,} customers** are classified as High Risk (churn probability ≥ 70%)")
            if total:
                try:
                    pct = h / int(total) * 100
                    lines.append(f"  — that's **{pct:.1f}%** of your total customer base")
                except Exception:
                    pass
        except Exception:
            lines.append(f"- High risk customers: {high}")
    if revenue_at_risk is not None:
        try:
            lines.append(f"- **Revenue at risk from this group:** ${float(revenue_at_risk):,.2f}")
        except Exception:
            pass
    # Add top drivers context
    shap = context.get("shap")
    features = _shortlist_top_features_from_shap(shap, top_n=3)
    if features:
        lines.append("\n**Key factors driving their risk:**")
        for f in features:
            name = f.get("name", "") if isinstance(f, dict) else str(f)
            lines.append(f"- {name}")
    lines.append(
        "\n**Recommended action:** Run a simulation in the **Simulate** tab to estimate "
        "the ROI of offering discounts or contract extensions to this high-risk cohort."
    )
    return "\n".join(lines)
def _handle_lifecycle(context: Dict[str, Any]) -> str:
    """Answer questions about lifecycle risk analysis results."""
    lifecycle = context.get("lifecycle") or {}
    if not lifecycle or lifecycle.get("error"):
        error_msg = lifecycle.get("error", "") if lifecycle else ""
        return (
            "Lifecycle risk analysis hasn't been run yet, or no suitable tenure column was found.\n\n"
            + (f"Reason: {error_msg}\n\n" if error_msg else "") +
            "The lifecycle analysis requires a numeric column whose name contains 'tenure', 'month', 'months', "
            "'duration', or 'time'. It then automatically buckets customers into:\n"
            "- **Early stage** (≤3 months)\n"
            "- **Mid stage** (4–12 months)\n"
            "- **Late stage** (>12 months)\n\n"
            "Use the **Segment Analysis** section and select your tenure column to run this."
        )
    col = lifecycle.get("column_used", "")
    highest = lifecycle.get("highest_risk_segment", "")
    segments = lifecycle.get("segments") or {}
    lines = ["**Customer Lifecycle Risk Analysis:**\n"]
    if col:
        lines.append(f"- **Column used:** {col}\n")
    def _fmt_stage(stage_name: str, label: str, bracket: str) -> str:
        s = segments.get(stage_name) or {}
        customers = s.get("customers", 0)
        churned = s.get("churned", 0)
        rate = s.get("churn_rate", 0)
        try:
            rate_pct = float(rate) * 100 if float(rate) <= 1 else float(rate)
            flag = " ⚠️ HIGHEST RISK" if stage_name == highest else ""
            return (
                f"- **{label}** ({bracket}): {rate_pct:.1f}% churn rate "
                f"({int(churned):,} of {int(customers):,} customers){flag}"
            )
        except Exception:
            return f"- {label}: customers={customers}, churned={churned}, rate={rate}"
    lines.append(_fmt_stage("early_stage", "Early Stage", "≤3 months"))
    lines.append(_fmt_stage("mid_stage", "Mid Stage", "4–12 months"))
    lines.append(_fmt_stage("late_stage", "Late Stage", ">12 months"))
    if highest:
        stage_label = {"early_stage": "Early Stage", "mid_stage": "Mid Stage", "late_stage": "Late Stage"}.get(highest, highest)
        lines.append(f"\n**Highest-risk segment: {stage_label}**")
        if highest == "early_stage":
            lines.append(
                "Early-tenure customers are most vulnerable — focus on onboarding quality, "
                "early engagement incentives, and proactive support in the first 3 months."
            )
        elif highest == "mid_stage":
            lines.append(
                "Mid-tenure customers show elevated risk — investigate service satisfaction, "
                "usage patterns, and whether contract or pricing issues are emerging."
            )
        else:
            lines.append(
                "Long-tenure customers are churning despite their history — "
                "this often reflects accumulated dissatisfaction. Loyalty programmes and account reviews are key."
            )
    return "\n".join(lines)
def _handle_data_stats(context: Dict[str, Any]) -> str:
    """Answer questions about numeric statistics and distributions in the dataset."""
    eda = context.get("eda") or {}
    if not eda:
        return (
            "No dataset statistics are available yet.\n\n"
            "Upload a dataset and click **Run Full EDA** to generate numeric summaries, "
            "correlations, and category distributions."
        )
    numeric_summary = eda.get("numeric_summary") or {}
    top_categories = eda.get("top_categories") or {}
    correlation = eda.get("correlation") or {}
    lines = ["**Dataset Statistics Summary:**\n"]
    if numeric_summary:
        lines.append(f"**Numeric columns ({len(numeric_summary)}):**")
        for col_name, stats in list(numeric_summary.items())[:8]:
            if isinstance(stats, dict):
                mean = stats.get("mean")
                mn = stats.get("min")
                mx = stats.get("max")
                parts = []
                if mean is not None:
                    try:
                        parts.append(f"mean={float(mean):.2f}")
                    except Exception:
                        pass
                if mn is not None:
                    try:
                        parts.append(f"min={float(mn):.2f}")
                    except Exception:
                        pass
                if mx is not None:
                    try:
                        parts.append(f"max={float(mx):.2f}")
                    except Exception:
                        pass
                stat_str = ", ".join(parts)
                lines.append(f"- **{col_name}**: {stat_str}" if stat_str else f"- **{col_name}**")
        lines.append("")
    if top_categories:
        lines.append(f"**Categorical columns ({len(top_categories)}):**")
        for col_name, vals in list(top_categories.items())[:5]:
            if isinstance(vals, dict):
                top_val = list(vals.items())[:3]
                top_str = ", ".join([f"{k} ({v:.1%})" if isinstance(v, float) else f"{k} ({v})" for k, v in top_val])
                lines.append(f"- **{col_name}**: top values — {top_str}")
        lines.append("")
    if correlation:
        num_corr_cols = len(correlation)
        lines.append(
            f"A **correlation matrix** is available for {num_corr_cols} numeric columns. "
            "Check the Full EDA section for the color-coded heatmap."
        )
    if len(lines) <= 1:
        return (
            "Full EDA stats aren't available yet.\n\n"
            "Click **Run Full EDA** in the EDA section to generate numeric summaries, "
            "distributions, and correlation matrices for your dataset."
        )
    return "\n".join(lines)
def _handle_training_details(context: Dict[str, Any]) -> str:
    """Answer detailed questions about the training process."""
    metrics = context.get("metrics") or {}
    history = context.get("model_history") or []
    if not metrics and not history:
        return (
            "No training has been done yet.\n\n"
            "Go to the **Train** tab:\n"
            "1. Select the **target column** (the churn label)\n"
            "2. Choose a **model** (Logistic Regression, Random Forest, XGBoost, etc.)\n"
            "3. Pick **Fast** or **Full** training mode\n"
            "4. Click **Train**\n\n"
            "The system supports 8 algorithms and automatically detects overfitting."
        )
    lines = ["**Training Details:**\n"]
    model_type = metrics.get("model_type") or context.get("model_type") or ""
    if model_type:
        lines.append(f"- **Active model:** {model_type}")
    mode = metrics.get("mode") or metrics.get("training_mode") or context.get("training_mode") or ""
    if mode:
        lines.append(f"- **Training mode:** {mode}")
    target = (context.get("eda") or {}).get("target_column") or metrics.get("target_col") or ""
    if target:
        lines.append(f"- **Target column:** {target}")
    fit_status = metrics.get("fit_status") or metrics.get("fit") or ""
    if fit_status:
        lines.append(f"- **Fit status:** {fit_status} — {'model generalizes well to new data' if 'Good' in str(fit_status) else 'check for overfitting'}")
    threshold = metrics.get("threshold") or 0.35
    lines.append(f"- **Prediction threshold:** {threshold} (lower than default 0.5 to maximize recall — catching more churners)")
    if history:
        suspicious_count = sum(1 for h in history if h.get("suspicious"))
        if suspicious_count:
            lines.append(
                f"- **{suspicious_count} model(s) flagged as suspicious** this session "
                "(all 4 metrics ≥98% — possible data leakage or trivially separable data)"
            )
    train_f1 = metrics.get("train_f1") or metrics.get("train_score")
    test_f1 = metrics.get("test_f1") or metrics.get("f1")
    if train_f1 and test_f1:
        try:
            gap = abs(float(train_f1) - float(test_f1)) * 100
            lines.append(f"- **Train/test F1 gap:** {gap:.2f}pp — {'✅ good generalization' if gap < 3 else '⚠️ some overfitting' if gap < 15 else '❌ significant overfitting'}")
        except Exception:
            pass
    return "\n".join(lines)
def _handle_full_summary(context: Dict[str, Any]) -> str:
    if not context:
        return (
            "No session results are available yet.\n\n"
            "Upload a dataset, run EDA, train a model, and run predictions to generate a complete churn summary."
        )
    eda = context.get("eda") or {}
    metrics = context.get("metrics") or {}
    predictions = context.get("predictions") or {}
    features = _shortlist_top_features_from_shap(context.get("shap"), top_n=5)
    lines: List[str] = ["**Dataset**"]
    n_rows = eda.get("n_rows") or (eda.get("shape") or [None])[0]
    n_cols = eda.get("n_cols") or (eda.get("shape") or [None, None])[1]
    cols = eda.get("columns")
    col_count = len(cols) if isinstance(cols, list) else n_cols
    target = eda.get("target_column") or metrics.get("target_col") or ""
    missing_total = eda.get("missing_total")
    if n_rows is not None:
        try:
            lines.append(f"- Rows: {int(n_rows):,}")
        except Exception:
            lines.append(f"- Rows: {n_rows}")
    if col_count is not None:
        lines.append(f"- Columns: {col_count}")
    if target:
        lines.append(f"- Target column: {target}")
    if missing_total is not None:
        try:
            mv = int(missing_total)
            lines.append(f"- Missing values: {mv:,}" if mv else "- Missing values: none")
        except Exception:
            lines.append(f"- Missing values: {missing_total}")
    if len(lines) == 1:
        lines.append("- Dataset summary is not available yet")
    lines.append("\n**Model**")
    model_type = _get_model_type(context)
    if model_type:
        lines.append(f"- Current model: {model_type}")
    if metrics:
        metric_summary = _metric_summary_for_answer(context)
        if metric_summary:
            lines.append(f"- Metrics: {metric_summary}")
        fit_status = metrics.get("fit_status") or metrics.get("fit")
        if fit_status:
            lines.append(f"- Fit status: {fit_status}")
    else:
        lines.append("- Model metrics are not available yet")
    lines.append("\n**Churn**")
    actual = _get_actual_churn_rate(context)
    predicted = _get_predicted_churn_rate(context)
    if actual is not None:
        lines.append(f"- Actual churn rate: {_fmt_pct(actual)}")
    else:
        lines.append("- Actual churn rate: not available yet")
    if predicted is not None:
        lines.append(f"- Predicted churn rate: {_fmt_pct(predicted)}")
    else:
        lines.append("- Predicted churn rate: not available yet")
    if actual is not None and predicted is not None:
        diff = (predicted - actual) * 100
        direction = "underestimates" if diff < 0 else "overestimates" if diff > 0 else "matches"
        lines.append(f"- Comparison: the model {direction} churn by {abs(diff):.1f} percentage points")
    lines.append("\n**Predictions**")
    if predictions:
        total = predictions.get("total_customers") or predictions.get("total")
        churn_count = predictions.get("churn_count")
        high = predictions.get("high_risk")
        medium = predictions.get("medium_risk")
        low = predictions.get("low_risk")
        revenue_at_risk = predictions.get("revenue_at_risk")
        if total is not None:
            try:
                lines.append(f"- Customers scored: {int(total):,}")
            except Exception:
                lines.append(f"- Customers scored: {total}")
        if churn_count is not None:
            try:
                lines.append(f"- Predicted churners: {int(churn_count):,}")
            except Exception:
                lines.append(f"- Predicted churners: {churn_count}")
        for label, value in (("High risk", high), ("Medium risk", medium), ("Low risk", low)):
            if value is not None:
                try:
                    lines.append(f"- {label}: {int(value):,}")
                except Exception:
                    lines.append(f"- {label}: {value}")
        if revenue_at_risk is not None:
            try:
                lines.append(f"- Revenue at risk: ${float(revenue_at_risk):,.2f}")
            except Exception:
                lines.append(f"- Revenue at risk: {revenue_at_risk}")
    else:
        lines.append("- Predictions have not been run yet")
    lines.append("\n**Drivers**")
    if features:
        for i, f in enumerate(features, start=1):
            name = f.get("name", "<unknown>") if isinstance(f, dict) else str(f)
            score = _as_float(f.get("mean_abs_shap")) if isinstance(f, dict) else None
            if score is not None:
                lines.append(f"{i}. {name} - influence score {score:.4f}")
            else:
                lines.append(f"{i}. {name}")
    else:
        lines.append("- Top churn drivers are not available yet")
    lines.append("\n**Verdict**")
    recall = metrics.get("recall") if isinstance(metrics, dict) else None
    if recall is not None:
        lines.append(f"- {_recall_verdict(recall)}")
    else:
        lines.append("- Recall is not available yet, so the model's ability to catch churners cannot be judged.")
    if actual is not None and predicted is not None and predicted < actual:
        lines.append("- The model is underestimating churn, so improving recall should be the next priority.")
    elif recall is not None and (_as_rate(recall) or 0) < 0.70:
        lines.append("- Next step: tune or compare models to improve recall.")
    else:
        lines.append("- Next step: use the high-risk list and top drivers to plan retention actions.")
    return "\n".join(lines)
def _handle_model_tracking(context: Dict[str, Any]) -> str:
    """Rule-based handler: which model, compare models, best model."""
    current_model = context.get("current_model") or (context.get("metrics") or {}).get("model_type") or ""
    history = context.get("model_history") or []
    if not current_model and not history:
        return (
            "No model has been trained yet in this session.\n\n"
            "Go to the **Train** tab, select a target column and an algorithm, then click Train. "
            "You can train multiple models and the system will compare them automatically."
        )
    lines = []
    if current_model:
        lines.append(f"**Currently active model:** {current_model}\n")
        explanation = MODEL_EXPLANATIONS.get(current_model)
        if explanation:
            lines.append(f"*Why this model:* {explanation}\n")
    if history and isinstance(history, list) and len(history) > 0:
        valid = [h for h in history if not h.get("suspicious")]
        suspicious = [h for h in history if h.get("suspicious")]
        lines.append(f"**All trained models this session ({len(history)} total):**\n")
        for h in history:
            name = h.get("model_type") or h.get("model") or "Unknown"
            m2 = h.get("metrics") or h
            roc = m2.get("roc_auc")
            recall = m2.get("recall")
            f1 = m2.get("f1")
            fit = m2.get("fit_status") or h.get("fit_status") or ""
            susp_flag = " 🚩 **[SUSPICIOUS — excluded]**" if h.get("suspicious") else ""
            parts = []
            if roc is not None:
                try: parts.append(f"ROC-AUC: {float(roc):.3f}")
                except Exception: pass
            if recall is not None:
                try:
                    r_pct = float(recall) * 100 if float(recall) <= 1 else float(recall)
                    parts.append(f"Recall: {r_pct:.1f}%")
                except Exception: pass
            if f1 is not None:
                try:
                    f_pct = float(f1) * 100 if float(f1) <= 1 else float(f1)
                    parts.append(f"F1: {f_pct:.1f}%")
                except Exception: pass
            if fit:
                parts.append(f"Fit: {fit}")
            detail = " | ".join(parts)
            lines.append(f"- **{name}**{susp_flag}" + (f" — {detail}" if detail else ""))
        if valid:
            try:
                best = max(valid, key=lambda h: float((h.get("metrics") or h).get("roc_auc") or 0))
                best_name = best.get("model_type") or best.get("model") or "Best model"
                lines.append(f"\n✅ **Best model by ROC-AUC:** {best_name}")
            except Exception:
                pass
        if suspicious:
            lines.append(
                f"\n⚠️ {len(suspicious)} model(s) flagged as suspicious (all 4 metrics ≥98%) — "
                "likely data leakage or trivially separable data. These are excluded from auto-selection."
            )
    return "\n".join(lines)
def _handle_model_explanation(context: Dict[str, Any]) -> str:
    """Rule-based handler: why this model, why random forest, why model used."""
    current_model = context.get("current_model") or (context.get("metrics") or {}).get("model_type") or ""
    if not current_model:
        return (
            "No model has been trained yet.\n\n"
            "Train a model in the **Train** tab and then ask again — "
            "I'll explain why that algorithm suits churn prediction for your dataset."
        )
    explanation = MODEL_EXPLANATIONS.get(current_model)
    if not explanation:
        explanation = "This model is suited for tabular classification tasks like churn prediction."
    lines = [
        f"**Why {current_model}?**\n",
        explanation,
        "",
        "**How it fits churn prediction:**",
        "Churn prediction involves tabular data with a mix of numeric and categorical features. "
        f"{current_model} handles this well by learning patterns from features like tenure, payment behavior, "
        "and usage frequency — the kinds of signals that typically drive customer decisions.",
    ]
    # Brief comparison with alternatives
    alternatives = [k for k in MODEL_EXPLANATIONS if k != current_model][:3]
    if alternatives:
        lines.append("\n**Brief comparison with alternatives:**")
        for alt in alternatives:
            alt_exp = MODEL_EXPLANATIONS[alt]
            # Just first sentence
            first_sentence = alt_exp.split(".")[0] + "."
            lines.append(f"- **{alt}**: {first_sentence}")
    return "\n".join(lines)
def _handle_overfitting(context: Dict[str, Any]) -> str:
    metrics = context.get("metrics") or {}
    if not metrics:
        return (
            "Overfitting cannot be checked yet because no model metrics are available.\n\n"
            "Overfitting means a model learns the training data too closely and performs worse on new data. "
            "Train a model in split mode so the system can compare train vs test performance."
        )
    fit_status = metrics.get("fit_status") or metrics.get("fit") or ""
    train_score = _as_float(metrics.get("train_score") or metrics.get("train_f1"))
    test_score = _as_float(metrics.get("test_score") or metrics.get("f1"))
    lines = [
        "Overfitting means the model looks strong on training data but does not generalize well to new customers.",
        "",
    ]
    model_type = _get_model_type(context)
    if model_type:
        lines.append(f"Current model: {model_type}.")
    if train_score is not None and test_score is not None:
        gap = (train_score - test_score) * 100
        if gap > 15:
            verdict = "severe overfitting"
        elif gap > 8:
            verdict = "overfitting"
        elif gap > 3:
            verdict = "mild overfitting"
        else:
            verdict = "no major overfitting signal"
        lines.append(
            f"In your model, train score is {_fmt_pct(_as_rate(train_score))} and test score is {_fmt_pct(_as_rate(test_score))}, "
            f"a gap of {gap:.1f} percentage points. That indicates {verdict}."
        )
    elif fit_status:
        lines.append(f"Your system's fit status is **{fit_status}**.")
    else:
        lines.append("Train/test comparison is not available, likely because the model was trained on the full dataset.")
    recall = _get_metric(context, "recall")
    if recall is not None:
        lines.append(f"Recall is {_fmt_pct(_as_rate(recall))}. {_recall_verdict(recall)}")
    lines.append("\nNext step: if overfitting is present, try a simpler model, stronger regularization, or more conservative tree settings; if recall is below your business target, tune for recall before deployment.")
    return "\n".join(lines)
# -------------------------
# Intent handler registry
# -------------------------
_AVAILABLE_MODELS_LIST = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "Decision Tree",
    "Gradient Boosting",
    "AdaBoost",
    "Naive Bayes",
    "CatBoost",
]
def _handle_available_models(context: Dict[str, Any]) -> str:
    models = context.get("available_models") or _AVAILABLE_MODELS_LIST
    trained = []
    history = context.get("model_history") or []
    for h in history:
        name = h.get("model_type") or h.get("model") or ""
        if name:
            trained.append(name)
    current = context.get("current_model") or (context.get("metrics") or {}).get("model_type") or ""
    lines = ["This system supports the following algorithms for training:\n"]
    for m in models:
        lines.append(f"- {m}")
    if trained:
        lines.append(f"\n**Models you have already trained:** {', '.join(trained)}")
    if current:
        lines.append(f"**Currently active model:** {current}")
    lines.append(
        "\nTo train a model, go to the **Train** tab, select an algorithm and target column, and start training."
    )
    return "\n".join(lines)
_INTENT_HANDLERS = {
    "help": _handle_help,
    "churn_rate": _handle_churn_rate,
    "top_drivers": _handle_top_drivers,
    "metrics": _handle_metrics,
    "simulation": _handle_simulation,
    "columns": _handle_columns,
    "explain_customer": _handle_explain_customer,
    "recommend_actions": _handle_recommend_actions,
    "full_summary": _handle_full_summary,
    # New handlers
    "dataset_summary": _handle_dataset_summary,
    "missing_values": _handle_missing_values,
    "model_info": _handle_model_info,
    "segment_analysis": _handle_segment_analysis,
    "predictions_summary": _handle_predictions_summary,
    "high_risk": _handle_high_risk,
    "lifecycle": _handle_lifecycle,
    "data_stats": _handle_data_stats,
    "training_details": _handle_training_details,
    "overfitting": _handle_overfitting,
    # model tracking & explanation (Feature 1 & 2)
    "model_tracking": _handle_model_tracking,
    "model_explanation": _handle_model_explanation,
    # available models
    "available_models": _handle_available_models,
}
# -------------------------
# Intent patterns & examples
# -------------------------
_INTENTS = [
    (re.compile(r"\b(help|what can you do|how to|commands)\b"), "help"),
    (re.compile(r"\b(churn rate|what is the churn|current churn|how many left|customers left|how many churn)\b"), "churn_rate"),
    (re.compile(r"\b(is churn (high|low|moderate)|churn (high|low|moderate)|churn level|churn risk level)\b"), "churn_rate"),
    (re.compile(r"\b(why (are )?customers churning|drivers of churn|reasons for churn|why people leave)\b"), "top_drivers"),
    (re.compile(r"\b(top features|feature importance|top 5 features|top drivers)\b"), "top_drivers"),
    (re.compile(r"\b(model metrics|how good is (the |my )?model|model (accuracy|performance|results)|accuracy score|precision score|recall score|f1 score|roc_auc|roc auc|show (me )?(the )?metrics)\b"), "metrics"),
    (re.compile(r"\b(simulation|roi|return on investment|revenue saved|action cost|how much do we save)\b"), "simulation"),
    (re.compile(r"\b(explain customer|explain user|explain row|explain id)\b"), "explain_customer"),
    (re.compile(r"\b(columns|what columns|list columns|schema|fields)\b"), "columns"),
    (re.compile(r"\b(what should i do|what actions? (should|can) (i|we) take|how to improve (retention|churn|sales)|improve retention|recommend(ed)? actions?|next steps?)\b"), "recommend_actions"),
    (re.compile(r"\b(show all|full summary|complete analysis|all results|everything|overview)\b"), "full_summary"),
    # New intent patterns
    (re.compile(r"\b(dataset|data summary|eda|eda summary|what data|what dataset|about (the )?data|describe (the )?data(set)?)\b"), "dataset_summary"),
    (re.compile(r"\b(how many rows|how many records|dataset size|how large|how big)\b"), "dataset_summary"),
    (re.compile(r"\b(missing (values?|data)|null values?|incomplete data|data quality)\b"), "missing_values"),
    # available models — must come BEFORE model_info to avoid wrong handler firing
    (re.compile(r"\b(what models? (can i|can we|do you|does (the )?system|are available|do i have|can be)|available models?|list (all )?models?|which models? (can i|are supported|exist)|models? (supported|available)|algorithms? (available|supported|can i use))\b"), "available_models"),
    # model_info — "which model" removed to avoid conflict with available_models
    (re.compile(r"\b(model info|trained model|what model|how many models|model(s)? trained)\b"), "model_info"),
    (re.compile(r"\b(segment|group analysis|category analysis|churn by|breakdown by|analyze by)\b"), "segment_analysis"),
    (re.compile(r"\b(predict(ion)?s?|how many (customers? )?(are )?predicted|predicted churn|scoring|scored)\b"), "predictions_summary"),
    (re.compile(r"\b(high risk|high.risk customers|at risk|most likely to churn|top risk)\b"), "high_risk"),
    (re.compile(r"\b(lifecycle|life.?cycle|early stage|mid stage|late stage|tenure stage|customer stage)\b"), "lifecycle"),
    (re.compile(r"\b(stats|statistics|distribution|numeric summary|data stats|correlation|mean value|average value|std deviation|show (me )?(the )?(stats|statistics|distribution|correlation))\b"), "data_stats"),
    (re.compile(r"\b(overfit|overfitting|underfit|underfitting|generaliz(e|ation)|train.?test gap|fit status)\b"), "overfitting"),
    (re.compile(r"\b(training (detail|info|process)|how (was|is) (the )?model trained|training mode|target column|which column)\b"), "training_details"),
    # model tracking & comparison — "which model" removed to avoid conflict with available_models
    (re.compile(r"\b(compare models?|best model|model comparison|models? trained)\b"), "model_tracking"),
    # model explanation (Feature 2)
    (re.compile(r"\b(why (this|the|use|using|random forest|logistic|xgboost|catboost|decision tree|naive bayes|gradient|adaboost) ?model|why model used|explain (the )?model choice|why (is|was) .* (model|algorithm) (used|chosen|selected))\b"), "model_explanation"),
]
_INTENT_EXAMPLES = {
    "help": ["what can you do", "help me", "how to use"],
    "churn_rate": ["what is the churn rate", "how many customers churn", "how many left"],
    "top_drivers": ["why are customers churning", "drivers of churn", "what causes churn"],
    "metrics": ["show model metrics", "what is the accuracy", "what is f1 score"],
    "simulation": ["what is the roi", "how much revenue saved", "simulate retention action"],
    "explain_customer": ["explain customer 123", "why did this customer churn"],
    "columns": ["list columns", "what columns are in the dataset"],
    "recommend_actions": ["what should i do to improve retention", "recommend actions to improve sales"],
    "full_summary": ["show all results", "full summary", "complete analysis", "give me an overview"],
    # New examples
    "dataset_summary": ["what does my dataset contain", "describe my data", "tell me about the dataset", "how many rows do i have"],
    "missing_values": ["are there missing values", "how many nulls", "is my data complete", "missing data"],
    "model_info": ["which model did i train", "what model is active", "how many models trained", "model information"],
    "segment_analysis": ["analyze churn by gender", "what is churn by segment", "churn by contract type", "group analysis"],
    "predictions_summary": ["how many customers are predicted to churn", "show predictions", "prediction results", "how many high risk"],
    "high_risk": ["show high risk customers", "who is most likely to churn", "high risk segment", "customers at risk"],
    "lifecycle": ["what is the lifecycle risk", "early stage churn", "late stage customers", "lifecycle analysis"],
    "data_stats": ["show data statistics", "what is the average", "show distributions", "numeric summary"],
    "training_details": ["how was the model trained", "what is the training mode", "which target column", "training process"],
    "overfitting": ["is my model overfitting", "train test gap", "fit status", "does the model generalize"],
    "model_tracking": ["which model is being used", "compare models", "best model", "show model comparison", "what models did i train"],
    "model_explanation": ["why this model", "why random forest", "why is logistic regression used", "explain the model choice", "why was xgboost selected"],
    "available_models": ["what models can i train", "list all models", "which algorithms are available", "what models does the system support"],
}
_SYNONYMS = {
    "left": ["churned", "cancelled", "left", "resigned"],
    "roi": ["return on investment", "return", "roi%"],
    "revenue": ["income", "sales", "revenue_saved", "saved"],
    "accuracy": ["acc", "precision", "recall", "f1"],
    "customer": ["user", "client", "subscriber"],
    "columns": ["schema", "fields", "features"],
    "simulate": ["simulation", "simulate", "what if"],
    "dataset": ["data", "file", "csv", "uploaded file"],
    "missing": ["null", "empty", "blank", "incomplete"],
    "model": ["algorithm", "classifier", "ml model"],
    "predict": ["score", "inference", "prediction"],
    "segment": ["group", "category", "breakdown", "split"],
    "lifecycle": ["tenure", "stage", "early", "late"],
}
def _apply_synonym_expansion(text: str) -> str:
    tokens = re.findall(r"\w+", (text or "").lower())
    for canon, aliases in _SYNONYMS.items():
        for i, tok in enumerate(tokens):
            if tok in aliases:
                tokens[i] = canon
    return " ".join(tokens)
def _fuzzy_intent_match(query: str) -> Tuple[str, float]:
    q_syn = _apply_synonym_expansion(_normalize(query))
    best_intent = ""
    best_score = 0.0
    for intent_key, examples in _INTENT_EXAMPLES.items():
        for ex in examples:
            ex_syn = _apply_synonym_expansion(_normalize(ex))
            score = difflib.SequenceMatcher(None, q_syn, ex_syn).ratio()
            if score > best_score:
                best_score = score
                best_intent = intent_key
    if SKLEARN_AVAILABLE:
        try:
            corpus = []
            mapping = []
            for intent_key, examples in _INTENT_EXAMPLES.items():
                for ex in examples:
                    corpus.append(ex)
                    mapping.append(intent_key)
            vectorizer = TfidfVectorizer().fit(corpus + [q_syn])
            corpus_vecs = vectorizer.transform(corpus)
            q_vec = vectorizer.transform([q_syn])
            sims = cosine_similarity(q_vec, corpus_vecs).flatten()
            idx = int(sims.argmax())
            tfidf_score = float(sims[idx])
            tfidf_intent = mapping[idx]
            if tfidf_score > best_score:
                best_score = tfidf_score
                best_intent = tfidf_intent
        except Exception:
            pass
    return best_intent, float(best_score)
# -------------------------
# LLM helpers (OpenAI)
# -------------------------
def _build_system_prompt() -> str:
    return (
        # ── ROLE ──────────────────────────────────────────────────────────────
        "You are a senior data analyst embedded inside a production customer churn platform. "
        "The user's dataset has been processed. EDA, model training, predictions, and optional "
        "segment/lifecycle/simulation analysis results are all provided in the Context below. "
        "Your job: give precise, honest, business-ready answers — every time.\n\n"
        # ── STEP 0: CLASSIFY FIRST ────────────────────────────────────────────
        "════════════════════════════════════════\n"
        "STEP 0 — CLASSIFY THE QUESTION BEFORE ANSWERING (MANDATORY)\n"
        "════════════════════════════════════════\n"
        "Every question falls into exactly one type. Identify it silently, then answer:\n\n"
        "TYPE 1 — DATA QUESTION (about this user's specific numbers, models, results):\n"
        "  → Use ONLY values from Context. Quote exact numbers. "
        "Never invent, estimate, or paraphrase a number. "
        "If a value is missing, say exactly: '[field] is not available yet — run [step] to generate it.'\n\n"
        "TYPE 2 — CONCEPTUAL QUESTION (what is recall? how does XGBoost work? what causes churn?):\n"
        "  → Answer from your expert ML knowledge. NEVER say 'not available' for general concepts. "
        "After explaining, connect to the user's actual Context numbers only briefly where relevant. "
        "Do not dump the full churn report for pure definitions.\n\n"
        "TYPE 3 — MIXED QUESTION (is my recall good? is the model overfitting?):\n"
        "  → Briefly explain the concept, then immediately apply it to the user's exact Context numbers.\n\n"
        # ── HARD RULES ────────────────────────────────────────────────────────
        "════════════════════════════════════════\n"
        "HARD RULES — VIOLATION = WRONG ANSWER\n"
        "════════════════════════════════════════\n\n"
        "RULE 1 — NEVER HALLUCINATE MODELS:\n"
        "The ONLY models that exist are those named in [CURRENT MODEL] and [MODEL HISTORY]. "
        "If neither contains a model name, say: 'No model has been trained yet.' "
        "Never invent or assume an algorithm name.\n\n"
        "RULE 2 — NEVER CONFUSE ACTUAL VS PREDICTED CHURN RATE:\n"
        "  ACTUAL churn rate  → source: [MODEL METRICS] → field: churn_rate\n"
        "                       This is ground truth from training labels. The ONLY valid source.\n"
        "                       Do NOT read actual churn rate from EDA — it is not stored there.\n"
        "  PREDICTED churn rate → source: [PREDICTION RESULTS] → field: predicted_churn_rate\n"
        "                         This is the model's forecast. Entirely different concept.\n"
        "  ALWAYS label both explicitly: 'Actual churn rate: X%' and 'Predicted churn rate: Y%'.\n"
        "  If both are present, ALWAYS state whether the model is underestimating or overestimating.\n"
        "  NEVER present predicted_churn_rate as the actual/observed churn rate.\n\n"
        "RULE 3 — RECALL DETERMINES MODEL QUALITY VERDICT (NO EXCEPTIONS):\n"
        "  Recall < 50%  -> MUST say: 'This model is missing many churners. "
        "It is not suitable for churn decisions in its current state.'\n"
        "                  FORBIDDEN: 'accurate', 'performs well', 'good model', 'suitable for deployment'.\n"
        "  Recall 50-69% -> MUST say: 'This model captures most churners, but recall can still improve.'\n"
        "  Recall >= 70%  -> Strong. The model is strong at detecting churners.\n"
        "  This rule overrides any positive signal from accuracy or ROC-AUC alone.\n\n"
        "RULE 4 — NEVER INVENT DATA:\n"
        "For TYPE 1 questions, if a Context field is missing, say so and name the step to run. "
        "Never substitute a plausible-sounding number.\n\n"
        "RULE 5 — NEVER CONTRADICT THE USER:\n"
        "If the user states a fact about their own actions, accept it and respond accordingly.\n\n"
        # ── MANDATORY CHECKLIST ───────────────────────────────────────────────
        "════════════════════════════════════════\n"
        "MANDATORY ANSWER CHECKLIST — MODEL / CHURN / PERFORMANCE QUESTIONS\n"
        "════════════════════════════════════════\n"
        "When answering ANY question about model quality, churn, predictions, or 'why is churn high', "
        "your response MUST include ALL of the following that are available in Context. "
        "If an item is not available, explicitly state it is missing.\n\n"
        "  [ ] 1. Actual churn rate — from [MODEL METRICS] churn_rate\n"
        "  [ ] 2. Predicted churn rate — from [PREDICTION RESULTS] predicted_churn_rate\n"
        "  [ ] 3. Model metrics — accuracy, recall, F1, ROC-AUC from [MODEL METRICS]\n"
        "  [ ] 4. Recall quality verdict — apply RULE 3 thresholds, give a clear judgment\n"
        "  [ ] 5. Top churn drivers — from [TOP CHURN DRIVERS], explain what features drive churn\n\n"
        "Answering with fewer than the available items above = incomplete answer. "
        "Do NOT answer model/churn questions using a single metric in isolation.\n\n"
        # ── MODEL HISTORY ─────────────────────────────────────────────────────
        "════════════════════════════════════════\n"
        "MODEL HISTORY — MANDATORY CONSULTATION\n"
        "════════════════════════════════════════\n"
        "- [MODEL HISTORY] contains every model trained, with full metrics.\n"
        "- For ANY model-related question, you MUST read [MODEL HISTORY] first.\n"
        "- NEVER say 'metrics not available' for a model whose numbers appear in [MODEL HISTORY].\n"
        "- When comparing models, use ALL entries with their exact listed numbers — skip none.\n"
        "- The active model is named in [CURRENT MODEL]. Use that exact name only.\n\n"
        # ── AVAILABLE MODELS ──────────────────────────────────────────────────
        "════════════════════════════════════════\n"
        "AVAILABLE VS TRAINED MODELS\n"
        "════════════════════════════════════════\n"
        "- [AVAILABLE MODELS] = what the system supports for training (can be trained).\n"
        "- [MODEL HISTORY]    = what the user has already trained.\n"
        "- 'What models can I use/try/train?' → answer from [AVAILABLE MODELS].\n"
        "- 'What have I trained / compare models?' → answer from [MODEL HISTORY].\n"
        "- Never confuse these two lists.\n\n"
        # ── RESPONSE STRUCTURE ────────────────────────────────────────────────
        "════════════════════════════════════════\n"
        "RESPONSE STRUCTURE — FOLLOW FOR EVERY ANSWER\n"
        "════════════════════════════════════════\n"
        "1. STATE   — Lead with the direct answer and exact numbers.\n"
        "2. EXPLAIN — What does this mean in plain business language?\n"
        "3. EVALUATE — Is this good, acceptable, or a problem? Why? "
        "Use RULE 3 for model quality. Use domain knowledge for concepts.\n"
        "4. SUGGEST — One concrete, actionable next step.\n\n"
        "Style rules (all mandatory):\n"
        "- Natural prose. No bullet points unless the user asks.\n"
        "- Lead with the answer — never open with 'Based on the context' or 'Great question'.\n"
        "- No 'I am an AI' disclaimers.\n"
        "- If the user is rude, stay calm and professional. Never lecture.\n"
        "- Concise: say exactly what is needed, nothing more."
    )
def _build_user_prompt(query: str, safe_context: Dict[str, Any]) -> str:
    """
    Build the full user-turn message sent to the LLM.
    Every available section of the session context is included so the model
    can reason over all results — exactly as if the user had uploaded the data
    directly into ChatGPT.
    """
    sections: List[str] = []
    # ── 1. Dataset / EDA ───────────────────────────────────────────────────
    eda = safe_context.get("eda")
    if eda:
        eda_lines = ["[DATASET & EDA]"]
        rows = eda.get("n_rows") or (eda.get("shape") or [None])[0]
        cols_count = eda.get("n_cols") or (eda.get("shape") or [None, None])[1]
        if rows:
            eda_lines.append(f"  rows: {rows}")
        if cols_count:
            eda_lines.append(f"  columns: {cols_count}")
        col_names = eda.get("columns")
        if col_names:
            listed = ", ".join(col_names[:40])
            suffix = f" ... (+{len(col_names) - 40} more)" if len(col_names) > 40 else ""
            eda_lines.append(f"  column names: {listed}{suffix}")
        target = eda.get("target_column")
        if target:
            eda_lines.append(f"  target column (churn label): {target}")
        missing = eda.get("missing_total")
        if missing is not None:
            eda_lines.append(f"  missing values total: {missing}")
        churn_rate = eda.get("churn_rate") or eda.get("target_churn_rate")
        if churn_rate is not None:
            eda_lines.append(
                f"  churn_rate (EDA): {churn_rate}"
                f"  ← NOTE: use [MODEL METRICS] churn_rate for the authoritative actual churn rate"
            )
        sections.append("\n".join(eda_lines))
    else:
        sections.append("[DATASET & EDA]\n  not available (dataset not yet uploaded or EDA not run)")
    # ── 2. Current model (always show explicitly) ──────────────────────────
    current_model = safe_context.get("current_model") or ""
    metrics = safe_context.get("metrics")
    if not current_model and metrics:
        current_model = metrics.get("model_type") or metrics.get("model") or ""
    if current_model:
        sections.append(
            f"[CURRENT MODEL]\n"
            f"  active model: {current_model}\n"
            f"  NOTE: This is the ONLY active model. Do not name any other model as current."
        )
    else:
        sections.append("[CURRENT MODEL]\n  not available (no model has been trained yet)")
    # ── 3. Model metrics ───────────────────────────────────────────────────
    if metrics:
        m_lines = ["[MODEL METRICS — metrics belong to the CURRENT MODEL listed above]"]
        # Actual churn rate may live here too (same concept as EDA churn_rate — ground truth)
        metrics_churn = metrics.get("churn_rate")
        if metrics_churn is not None:
            m_lines.append(
                f"  ACTUAL churn rate (authoritative — from training labels): {metrics_churn}"
                f"  ← THIS is the real churn rate. Use ONLY this value when answering 'what is the churn rate?'"
            )
        for key in ("accuracy", "recall", "precision", "f1", "roc_auc"):
            if key in metrics:
                m_lines.append(f"  {key}: {metrics[key]}")
        fit_status = metrics.get("fit_status") or metrics.get("fit")
        if fit_status:
            m_lines.append(f"  fit status: {fit_status}")
        mode = metrics.get("mode") or metrics.get("training_mode")
        if mode:
            m_lines.append(f"  training mode: {mode}")
        threshold = metrics.get("threshold")
        if threshold is not None:
            m_lines.append(f"  prediction threshold: {threshold}")
        train_f1 = metrics.get("train_f1") or metrics.get("train_score")
        test_f1 = metrics.get("test_f1") or metrics.get("f1")
        if train_f1 and test_f1:
            m_lines.append(f"  train F1: {train_f1}  |  test F1: {test_f1}")
        suspicious = metrics.get("suspicious")
        if suspicious is not None:
            m_lines.append(f"  suspicious (near-perfect, possible data leakage): {suspicious}")
        sections.append("\n".join(m_lines))
    else:
        sections.append("[MODEL METRICS]\n  not available (no model trained yet)")
    # ── 4. Model history / comparison ──────────────────────────────────────
    history = safe_context.get("model_history")
    if history:
        h_lines = [
            f"[MODEL HISTORY — {len(history)} model(s) trained]",
            "  IMPORTANT: Only these models have been trained. Do not mention any model not listed here.",
            "  ALL entries below have valid metrics — never say a model lacks metrics if it appears here."
        ]
        for h in history:
            name = h.get("model_type") or h.get("model") or "unknown"
            susp = h.get("suspicious", False)
            m2 = h.get("metrics") or {}
            roc = m2.get("roc_auc", "n/a")
            recall = m2.get("recall", "n/a")
            f1 = m2.get("f1", "n/a")
            acc = m2.get("accuracy", "n/a")
            flag = " [SUSPICIOUS — possible data leakage, excluded from auto-selection]" if susp else ""
            h_lines.append(
                f"  - {name}: ROC-AUC={roc}, recall={recall}, F1={f1}, accuracy={acc}{flag}"
            )
        h_lines.append(
            "  MANDATORY: When comparing models, use ALL entries above with their exact numbers."
        )
        sections.append("\n".join(h_lines))
    else:
        sections.append(
            "[MODEL HISTORY]\n"
            "  not available\n"
            "  IMPORTANT: Do not mention or compare any model if no history is listed here."
        )
    # ── 5. SHAP / feature importance ───────────────────────────────────────
    shap = safe_context.get("shap")
    if shap:
        top = shap.get("top_features") or []
        if top:
            s_lines = [f"[TOP CHURN DRIVERS — SHAP importance, top {len(top[:15])} features]"]
            for f in top[:15]:
                name = f.get("name", "?")
                score = f.get("mean_abs_shap")
                score_str = f"{score:.4f}" if score is not None else "n/a"
                s_lines.append(f"  - {name}: mean_abs_shap={score_str}")
            sections.append("\n".join(s_lines))
    else:
        sections.append("[SHAP / FEATURE IMPORTANCE]\n  not available (train a model to compute SHAP)")
    # ── 6. Predictions ─────────────────────────────────────────────────────
    predictions = safe_context.get("predictions")
    if predictions:
        p_lines = [
            "[PREDICTION RESULTS — model's forecast on the dataset]",
            "  NOTE: 'predicted_churn_rate' below is the MODEL'S PREDICTION, "
            "not the historical actual churn rate from the dataset. Do NOT confuse these two."
        ]
        for key in ("total_customers", "total", "predicted_churn_rate", "predicted_churn",
                    "high_risk", "medium_risk", "low_risk", "revenue_at_risk",
                    "churn_count", "churn_rate"):
            if key in predictions:
                label = key
                if key in ("predicted_churn_rate", "churn_rate"):
                    label = f"{key} (PREDICTED by model)"
                p_lines.append(f"  {label}: {predictions[key]}")
        sections.append("\n".join(p_lines))
    else:
        sections.append("[PREDICTION RESULTS]\n  not available (run predictions after training)")
    # ── 7. Simulation ──────────────────────────────────────────────────────
    sim = safe_context.get("simulate")
    if sim:
        sim_lines = ["[RETENTION SIMULATION RESULTS]"]
        for key in ("before_churn_rate", "after_churn_rate", "retained_customers",
                    "revenue_saved", "action_cost", "roi"):
            if key in sim:
                sim_lines.append(f"  {key}: {sim[key]}")
        sections.append("\n".join(sim_lines))
    else:
        sections.append("[SIMULATION RESULTS]\n  not available (run a simulation in the Simulate tab)")
    # ── 8. Lifecycle risk ──────────────────────────────────────────────────
    lifecycle = safe_context.get("lifecycle")
    if lifecycle:
        lc_lines = ["[LIFECYCLE RISK ANALYSIS]"]
        lc_lines.append(f"  column used: {lifecycle.get('column_used', 'n/a')}")
        lc_lines.append(f"  highest risk segment: {lifecycle.get('highest_risk_segment', 'n/a')}")
        segments = lifecycle.get("segments")
        if isinstance(segments, dict):
            for stage, stats in segments.items():
                if isinstance(stats, dict):
                    rate = stats.get("churn_rate", "n/a")
                    customers = stats.get("customers", "n/a")
                    lc_lines.append(f"  {stage}: churn_rate={rate}, customers={customers}")
        sections.append("\n".join(lc_lines))
    else:
        sections.append("[LIFECYCLE RISK]\n  not available (run lifecycle analysis)")
    # ── 9. Segment analysis ────────────────────────────────────────────────
    segment = safe_context.get("segment")
    if segment:
        seg_lines = ["[SEGMENT ANALYSIS RESULTS]"]
        col = segment.get("column", "n/a")
        mode = segment.get("mode", "")
        seg_lines.append(f"  column analyzed: {col}  |  mode: {mode}")
        segs = segment.get("segments") or []
        for s in segs[:10]:
            val = s.get("value", "?")
            rate = s.get("churn_rate", "n/a")
            customers = s.get("customers", "n/a")
            churned = s.get("churned", "n/a")
            seg_lines.append(f"  - {val}: churn_rate={rate}, customers={customers}, churned={churned}")
        sections.append("\n".join(seg_lines))
    else:
        sections.append("[SEGMENT ANALYSIS]\n  not available (run segment analysis)")
    # ── 10. Available models (what the system supports) ───────────────────
    available_models = safe_context.get("available_models") or []
    if available_models:
        av_lines = [
            "[AVAILABLE MODELS — algorithms this system supports for training]",
            "  NOTE: These are models the user CAN train, not necessarily what they have trained.",
            "  Use this section when the user asks 'what models can I use/train/try?'."
        ]
        for m in available_models:
            av_lines.append(f"  - {m}")
        sections.append("\n".join(av_lines))
    # ── Assemble final prompt ──────────────────────────────────────────────
    context_block = "\n\n".join(sections)
    return (
        f"Context:\n"
        f"{'=' * 60}\n"
        f"{context_block}\n"
        f"{'=' * 60}\n\n"
        f"User question: {redact_text(query)}\n\n"
        f"INSTRUCTION: First classify this question:\n"
        f"- If it asks about THIS USER'S specific data, models, or results → use ONLY the Context above. "
        f"Never invent numbers. If a value is 'not available', say so and guide the user.\n"
        f"- If it asks about an ML/analytics concept (overfitting, recall, AUC, etc.) → answer from "
        f"expert knowledge, then connect to the user's Context data where relevant.\n"
        f"- If mixed → explain the concept, then apply it to the Context data with exact numbers.\n"
        f"For churn rate: ACTUAL comes only from [MODEL METRICS] churn_rate. "
        f"PREDICTED comes only from [PREDICTION RESULTS] predicted_churn_rate. Always label which is which.\n"
        f"Follow the reasoning structure: state → explain → evaluate → suggest. Be natural and direct."
    )
def _call_openai_chat(system_prompt: str, user_prompt: str, model: str, max_tokens: int = 800, temperature: float = 1.0, timeout: int = 180) -> str:
    if not OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI SDK not available. Install `openai` to enable LLM mode.")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment. (Did you add load_dotenv() to app.py?)")
    client = OpenAI(api_key=api_key, timeout=float(timeout))
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=max_tokens,
        )
        
    except Exception as e:
        print(f"!!!!!!!!!! OPENAI API CALL FAILED !!!!!!!!!!")
        print(f"Error type: {type(e).__name__}, Error: {e}")
        raise e
    if resp and getattr(resp, "choices", None):
        content = resp.choices[0].message.content
        return content.strip()
    return "LLM returned no content."
# -------------------------
# Fallback helper for open questions
# -------------------------
def _fallback_for_open_question(query: str, context: Dict[str, Any]) -> str:
    q = (query or "").lower()
    if "improve" in q or "what should i do" in q or "recommend" in q:
        return _handle_recommend_actions(context or {})
    if "dataset" in q or "data" in q or "rows" in q or "upload" in q:
        return _handle_dataset_summary(context or {})
    if "missing" in q or "null" in q:
        return _handle_missing_values(context or {})
    if "predict" in q or "scoring" in q:
        return _handle_predictions_summary(context or {})
    if "segment" in q or "group" in q or "category" in q:
        return _handle_segment_analysis(context or {})
    if "lifecycle" in q or "tenure" in q or "stage" in q:
        return _handle_lifecycle(context or {})
    if "overfit" in q or "underfit" in q or "train test" in q or "fit status" in q:
        return _handle_overfitting(context or {})
    if "model" in q or "train" in q or "algorithm" in q:
        return _handle_model_info(context or {})
    return (
        "I'm not fully sure how to answer that with the current data.\n\n"
        "Here are a few things you can try:\n"
        "- 'What is the churn rate?'\n"
        "- 'Why are customers churning?'\n"
        "- 'How good is the model?'\n"
        "- 'What actions should we take?'\n"
        "- 'What does my dataset contain?'\n"
        "- 'How many customers are predicted to churn?'\n\n"
        "If you need broader analysis, enable the AI assistant (LLM toggle) for more flexible answers."
    )
# -------------------------
# Public API
# -------------------------
def respond_to_query(query: str, context: Dict[str, Any], use_llm: bool = False, llm_provider: str = "openai", llm_model: str = "gpt-4o-mini") -> str:
    """
    Main entrypoint for the chatbot.
    Control flow (strict separation — LLM is always checked first):
      1. PII guard — always applied first.
      2. If use_llm is True:
           → build full redacted context
           → call LLM
           → return LLM answer                          ← happy path
         If LLM fails for ANY reason (quota, network, timeout, auth):
           → fall through to rule-based system
           → prepend a one-line soft notice so the user knows
             (the app NEVER crashes or shows an error page)
      3. Rule-based system (use_llm False, or LLM fell through):
           a. Precise regex intent matching
           b. Fuzzy intent matching (difflib + TF-IDF)
           c. Final keyword fallback
    """
    if not query or not isinstance(query, str):
        return "It looks like your message was empty. Try asking something like 'What is the churn rate?' or 'Show me the top churn factors.'"
    # ── PII guard (always applied) ──────────────────────────────────────────
    if (re.search(r"[\w\.-]+@[\w\.-]+\.\w+", query)
            or re.search(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b", query)
            or re.search(r"\b\d{10}\b", query)):
        return (
            "For privacy reasons, I can't process personal identifiers like email addresses, "
            "phone numbers, or ID numbers in chat.\n\n"
            "To look up a specific customer, use the **Explain** tab and enter their row number from the dataset."
        )
    # ── BRANCH A: LLM mode ─────────────────────────────────────────────────
    # Exact factual handlers run before the LLM; open-ended questions still use LLM mode.
    # On ANY failure the system falls through to Branch B — the app never fails.
    # Exact data/hybrid intents are answered deterministically even when LLM mode
    # is enabled. The LLM can explain broad concepts, but handlers own facts.
    q_low = _normalize(query)
    if _is_concept_query(query):
        return _handle_concept_question(query, context or {})
    if use_llm:
        for pattern, intent_key in _INTENTS:
            if pattern.search(q_low) and _is_data_intent(intent_key):
                handler = _INTENT_HANDLERS.get(intent_key)
                if handler:
                    try:
                        answer = handler(context or {})
                        return _enforce_answer_completeness(answer, query, context or {}, intent_key)
                    except Exception:
                        return "Something went wrong while generating that answer. Try rephrasing your question."
        intent_key, score = _fuzzy_intent_match(query)
        if intent_key and score >= 0.75 and _is_data_intent(intent_key):
            handler = _INTENT_HANDLERS.get(intent_key)
            if handler:
                try:
                    answer = handler(context or {})
                    return _enforce_answer_completeness(answer, query, context or {}, intent_key)
                except Exception:
                    pass
    _prepend_notice = ""   # set only when LLM falls back
    if use_llm:
        try:
            safe_ctx = redact_context(context or {})
            system_prompt = _build_system_prompt()
            user_prompt = _build_user_prompt(query, safe_ctx)
            if llm_provider.lower() != "openai":
                return "That LLM provider isn't supported yet. Please use the default OpenAI option."
            answer = _call_openai_chat(system_prompt, user_prompt, model=llm_model)
            answer = redact_text(answer)
            return _enforce_answer_completeness(answer, query, context or {})   # success: return here, never touch rule-based
        except Exception as e:
            # ── Determine the most helpful one-liner for the notice ──────────
            err_str = str(e)
            if "429" in err_str or "insufficient_quota" in err_str or "quota" in err_str.lower():
                notice = (
                    "*(AI assistant unavailable — API quota reached. "
                    "Turn off the LLM toggle for uninterrupted access. "
                    "Showing rule-based answer below.)*\n\n"
                )
            elif "auth" in err_str.lower() or "api_key" in err_str.lower() or "401" in err_str:
                notice = (
                    "*(AI assistant unavailable — API key issue. "
                    "Showing rule-based answer below.)*\n\n"
                )
            else:
                notice = (
                    "*(AI assistant encountered a temporary issue — "
                    "showing rule-based answer below.)*\n\n"
                )
            _prepend_notice = notice
            # fall through to Branch B — use_llm stays True but we skip the if block now
    # ── BRANCH B: Rule-based system (logic unchanged from original) ─────────
    q_low = _normalize(query)
    # 1) Precise regex intent matching
    for pattern, intent_key in _INTENTS:
        if pattern.search(q_low):
            handler = _INTENT_HANDLERS.get(intent_key)
            if handler:
                try:
                    answer = handler(context or {})
                    answer = _enforce_answer_completeness(answer, query, context or {}, intent_key)
                    return _prepend_notice + answer
                except Exception:
                    return _prepend_notice + "Something went wrong while generating that answer. Try rephrasing your question."
    # 2) Fuzzy matching (difflib + optional TF-IDF)
    intent_key, score = _fuzzy_intent_match(query)
    if intent_key and score >= 0.6:
        handler = _INTENT_HANDLERS.get(intent_key)
        if handler:
            try:
                resp = handler(context or {})
                resp = _enforce_answer_completeness(resp, query, context or {}, intent_key)
                return _prepend_notice + resp + f"\n\n*(Matched as: {intent_key}, confidence: {score:.0%})*"
            except Exception:
                pass
    # 3) Final keyword fallback
    if "recommend" in q_low or "what should i do" in q_low or "improve" in q_low:
        answer = _handle_recommend_actions(context or {})
        return _prepend_notice + _enforce_answer_completeness(answer, query, context or {}, "recommend_actions")
    return _prepend_notice + (
        "I'm not fully sure about that one. Here are some things I can help with:\n\n"
        "- 'What does my dataset contain?'\n"
        "- 'Are there missing values?'\n"
        "- 'What is the churn rate?'\n"
        "- 'Why are customers churning?'\n"
        "- 'How good is the model?' / 'Which model did I train?'\n"
        "- 'How many customers are predicted to churn?'\n"
        "- 'What was the ROI of the last simulation?'\n"
        "- 'Show me all results'\n\n"
        "You can also enable the **AI assistant toggle** for broader, more flexible answers."
    )
