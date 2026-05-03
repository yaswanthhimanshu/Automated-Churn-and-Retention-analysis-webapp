#chatbot.py
from typing import Dict, Any, List, Tuple
import re
import difflib
import os
import json
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
        "best model", "best algorithm", "which model", "which algorithm",
        "model should", "algorithm should", "should i use",
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
        "means", "mean", "explain", "tell me about",
    )
    return any(marker in q for marker in concept_markers) and any(term in q for term in concept_terms)
def _is_full_summary_query(query: str) -> bool:
    q = _normalize(query)
    return any(marker in q for marker in ("show all", "full summary", "complete analysis", "all results", "everything", "overview"))
def _is_model_advice_query(query: str) -> bool:
    q = _normalize(query)
    advice_markers = (
        "which model is best", "what model is best", "best model for",
        "which algorithm is best", "what algorithm is best", "best algorithm for",
        "what model should", "which model should", "what algorithm should",
        "which algorithm should", "model should i use", "algorithm should i use",
        "best model to", "best algorithm to",
    )
    return any(marker in q for marker in advice_markers)
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
        "underfitting",
        "full_summary",
        "lifecycle",
        "data_stats",
        "available_models",
    }
_STRICT_HANDLER_INTENTS = {
    "help",
    "churn_rate",
    "metrics",
    "dataset_summary",
    "missing_values",
    "predictions_summary",
    "high_risk",
    "simulation",
    "columns",
    "explain_customer",
    "model_info",
    "model_tracking",
    "training_details",
    "full_summary",
    "lifecycle",
    "data_stats",
    "available_models",
}
_GROUNDED_HANDLER_INTENTS = {
    "top_drivers",
    "recommend_actions",
    "segment_analysis",
    "model_explanation",
    "overfitting",
    "underfitting",
}
_PRE_LLM_STRICT_INTENTS = {
    "help",
    "churn_rate",
    "metrics",
    "dataset_summary",
    "missing_values",
    "predictions_summary",
    "high_risk",
    "simulation",
    "columns",
    "explain_customer",
    "model_info",
    "model_tracking",
    "training_details",
    "full_summary",
    "lifecycle",
    "data_stats",
    "available_models",
}
_GENERAL_CHAT_MARKERS = (
    "who are you",
    "what are you",
    "how are you",
    "how intelligent are you",
    "how smart are you",
    "are you intelligent",
    "are you smart",
    "tell me about yourself",
    "what can you answer",
)
def _match_intent(query: str, fuzzy_threshold: float | None = None) -> Tuple[str, float]:
    q = _normalize(query)
    for pattern, intent_key in _INTENTS:
        if pattern.search(q):
            return intent_key, 1.0
    if fuzzy_threshold is not None:
        intent_key, score = _fuzzy_intent_match(query)
        if intent_key and score >= fuzzy_threshold:
            return intent_key, score
    return "", 0.0
def _is_general_chat_query(query: str) -> bool:
    q = _normalize(query)
    return any(marker in q for marker in _GENERAL_CHAT_MARKERS)
def _answer_from_handler(
    intent_key: str,
    query: str,
    context: Dict[str, Any],
    score: float | None = None,
) -> str:
    handler = _INTENT_HANDLERS.get(intent_key)
    if not handler:
        return ""
    answer = handler(context or {})
    if score is not None and score < 1.0:
        answer += f"\n\n*(Matched as: {intent_key}, confidence: {score:.0%})*"
    return answer
def _needs_completeness_guard(query: str, intent_key: str = "") -> bool:
    if _is_concept_query(query):
        return False
    if intent_key == "full_summary" or _is_full_summary_query(query):
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
def _has_system_results(context: Dict[str, Any]) -> bool:
    context = context or {}
    return any(
        bool(context.get(key))
        for key in ("eda", "metrics", "predictions", "shap", "simulate", "segment_result", "lifecycle")
    )
def _is_system_related_query(query: str) -> bool:
    q = _normalize(query)
    if not q:
        return False
    markers = (
        "churn", "customer", "retention", "dataset", "data", "eda", "missing",
        "model", "metric", "accuracy", "recall", "precision", "f1", "auc",
        "roc", "prediction", "predict", "risk", "driver", "feature", "shap",
        "segment", "lifecycle", "simulation", "roi", "revenue", "campaign",
        "result", "results", "outcome", "insight", "conclusion", "analysis",
        "system", "assistant", "chatbot", "dashboard", "webapp", "app", "section", "train", "trained", "deploy",
        "trust", "reliable", "good", "bad", "improve", "reduce", "action",
    )
    return any(marker in q for marker in markers)
def _append_if(lines: List[str], value: str | None) -> None:
    if value:
        lines.append(value)
def _dataset_fact(context: Dict[str, Any]) -> str:
    eda = context.get("eda") or {}
    if not isinstance(eda, dict) or not eda:
        return ""
    rows = eda.get("n_rows") or (eda.get("shape") or [None])[0]
    cols = eda.get("n_cols") or (eda.get("shape") or [None, None])[1]
    target = eda.get("target_column") or (context.get("metrics") or {}).get("target_col")
    missing = eda.get("missing_total")
    parts = []
    if rows is not None:
        try:
            parts.append(f"{int(rows):,} rows")
        except Exception:
            parts.append(f"{rows} rows")
    if cols is not None:
        parts.append(f"{cols} columns")
    if target:
        parts.append(f"target `{target}`")
    if missing is not None:
        try:
            parts.append("no missing values" if int(missing) == 0 else f"{int(missing):,} missing values")
        except Exception:
            parts.append(f"{missing} missing values")
    return "Dataset: " + ", ".join(parts) + "." if parts else ""
def _model_fact(context: Dict[str, Any]) -> str:
    model_type = _get_model_type(context)
    metric_summary = _metric_summary_for_answer(context)
    metrics = context.get("metrics") or {}
    if not model_type and not metric_summary:
        return ""
    sentence = f"Model: {model_type or 'trained model'}"
    if metric_summary:
        sentence += f" with {metric_summary}"
    fit_status = metrics.get("fit_status") if isinstance(metrics, dict) else None
    if fit_status:
        sentence += f"; fit status: {fit_status}"
    return sentence + "."
def _churn_fact(context: Dict[str, Any]) -> str:
    actual = _get_actual_churn_rate(context)
    predicted = _get_predicted_churn_rate(context)
    if actual is None and predicted is None:
        return ""
    if actual is not None and predicted is not None:
        diff = (predicted - actual) * 100
        direction = "underestimates" if diff < 0 else "overestimates" if diff > 0 else "matches"
        return (
            f"Churn: actual churn is {_fmt_pct(actual)} and predicted churn is {_fmt_pct(predicted)}, "
            f"so the model {direction} churn by {abs(diff):.1f} percentage points."
        )
    if actual is not None:
        return f"Churn: actual churn is {_fmt_pct(actual)}; predicted churn is not available yet."
    return f"Churn: predicted churn is {_fmt_pct(predicted)}; actual churn is not available yet."
def _prediction_fact(context: Dict[str, Any]) -> str:
    predictions = context.get("predictions") or {}
    if not isinstance(predictions, dict) or not predictions:
        return ""
    parts = []
    total = predictions.get("total_customers") or predictions.get("total")
    churn_count = predictions.get("churn_count") or predictions.get("predicted_churn")
    high = predictions.get("high_risk")
    revenue = predictions.get("revenue_at_risk")
    if total is not None:
        try:
            parts.append(f"{int(total):,} customers scored")
        except Exception:
            parts.append(f"{total} customers scored")
    if churn_count is not None:
        try:
            parts.append(f"{int(churn_count):,} predicted churners")
        except Exception:
            parts.append(f"{churn_count} predicted churners")
    if high is not None:
        try:
            parts.append(f"{int(high):,} high-risk customers")
        except Exception:
            parts.append(f"{high} high-risk customers")
    if revenue is not None:
        try:
            parts.append(f"${float(revenue):,.2f} revenue at risk")
        except Exception:
            parts.append(f"{revenue} revenue at risk")
    return "Predictions: " + ", ".join(parts) + "." if parts else ""
def _drivers_fact(context: Dict[str, Any]) -> str:
    features = _top_feature_names(context, top_n=3)
    if not features:
        return ""
    return "Drivers: the strongest available churn drivers are " + ", ".join(features) + "."
def _section_fact(context: Dict[str, Any], section: str) -> str:
    if section == "dataset":
        return _dataset_fact(context)
    if section == "model":
        return _model_fact(context)
    if section == "churn":
        return _churn_fact(context)
    if section == "predictions":
        return _prediction_fact(context)
    if section == "drivers":
        return _drivers_fact(context)
    if section == "simulation":
        sim = context.get("simulate") or {}
        if not isinstance(sim, dict) or not sim:
            return ""
        pieces = []
        if sim.get("roi") is not None:
            pieces.append(f"ROI {sim.get('roi')}")
        if sim.get("revenue_saved") is not None:
            pieces.append(f"revenue saved {sim.get('revenue_saved')}")
        if sim.get("retained_customers") is not None:
            pieces.append(f"retained customers {sim.get('retained_customers')}")
        return "Simulation: " + ", ".join(pieces) + "." if pieces else ""
    if section == "segments":
        segment = context.get("segment_result") or context.get("segment") or {}
        lifecycle = context.get("lifecycle") or {}
        pieces = []
        if isinstance(segment, dict) and segment:
            col = segment.get("column") or segment.get("result", {}).get("column")
            if col:
                pieces.append(f"segment analysis is available for {col}")
        if isinstance(lifecycle, dict) and lifecycle:
            high = lifecycle.get("highest_risk_segment")
            col = lifecycle.get("column_used")
            if high:
                pieces.append(f"highest lifecycle risk segment is {high}" + (f" using {col}" if col else ""))
        return "Segments: " + "; ".join(pieces) + "." if pieces else ""
    return ""
def _relevant_sections(query: str) -> List[str]:
    q = _normalize(query)
    if any(k in q for k in ("all result", "everything", "overview", "summary", "section", "outcome", "conclusion", "insight", "dashboard", "webapp", "app")):
        return ["dataset", "model", "churn", "predictions", "drivers", "segments", "simulation"]
    sections: List[str] = []
    if any(k in q for k in ("dataset", "data", "eda", "row", "column", "missing", "quality")):
        sections.append("dataset")
    if any(k in q for k in ("model", "metric", "accuracy", "recall", "precision", "f1", "auc", "roc", "overfit", "underfit", "train", "trust", "reliable", "deploy", "good", "bad")):
        sections.extend(["model", "churn"])
    if any(k in q for k in ("churn", "rate", "actual", "predicted")):
        sections.append("churn")
    if any(k in q for k in ("predict", "prediction", "risk", "customer", "score")):
        sections.append("predictions")
    if any(k in q for k in ("why", "driver", "factor", "cause", "feature", "shap", "reduce", "retention", "action", "improve")):
        sections.append("drivers")
    if any(k in q for k in ("segment", "group", "lifecycle", "stage", "tenure")):
        sections.append("segments")
    if any(k in q for k in ("simulation", "roi", "campaign", "revenue", "cost", "save")):
        sections.append("simulation")
    if not sections:
        sections = ["churn", "model", "drivers", "predictions"]
    unique: List[str] = []
    for section in sections:
        if section not in unique:
            unique.append(section)
    return unique
def _build_system_aware_answer(query: str, context: Dict[str, Any]) -> str:
    context = context or {}
    q = _normalize(query)
    if _is_concept_query(query):
        return _handle_concept_question(query, context)
    if _is_full_summary_query(query):
        return _handle_full_summary(context)
    if _is_model_advice_query(query):
        return _handle_model_advice_question(context)
    if not _has_system_results(context):
        return _general_missing_system_answer(query, context)
    facts = [
        fact for fact in (_section_fact(context, section) for section in _relevant_sections(query))
        if fact
    ]
    if not facts:
        facts = [
            fact for fact in (
                _churn_fact(context),
                _model_fact(context),
                _drivers_fact(context),
                _prediction_fact(context),
            )
            if fact
        ]
    if any(k in q for k in ("reduce", "improve", "retain", "retention", "action", "what should")):
        lead = "To reduce churn, start with the customers the model marks as risky and target the drivers behind that risk."
        close = "Next step: use the high-risk list for targeting, then design retention actions around the strongest drivers instead of treating all customers the same."
    elif "predict" in q and any(k in q for k in ("actual", "lower", "higher", "underestimate", "overestimate", "different")):
        lead = "The prediction result is different from actual churn because the model is estimating churn from learned patterns, not copying the historical label distribution."
        close = "Next step: check recall, calibration, and the prediction threshold before using the gap as a business forecast."
    elif any(k in q for k in ("dashboard", "webapp", "app", "system", "assistant", "chatbot")) and not any(k in q for k in ("trust", "reliable", "deploy", "good", "bad")):
        lead = "Here's what your current results show: the app is turning your dataset into churn risk, model quality, drivers, and action guidance."
        close = "Next step: read the results in this order: dataset quality, model quality, churn gap, high-risk customers, drivers, then retention action."
    elif any(k in q for k in ("trust", "reliable", "deploy", "good", "bad")):
        recall = _get_metric(context, "recall")
        lead = _recall_verdict(recall) if recall is not None else "I would not judge deployment readiness yet because recall is not available."
        close = "Next step: validate recall, train-test gap, and the high-risk list before using the model for business decisions."
    elif any(k in q for k in ("conclusion", "insight", "outcome", "think", "takeaway")):
        lead = "The main takeaway is that your webapp has enough churn outputs to support decisions, but the quality of the decision depends on model recall and the available churn drivers."
        close = "Next step: turn the top drivers into retention actions and keep recall as the main quality gate."
    elif any(k in q for k in ("why", "reason", "cause")):
        lead = "The best explanation should come from the model drivers, not from guessing."
        close = "Next step: focus analysis on the top drivers and validate them with segment or lifecycle views if available."
    else:
        lead = "Based on your current model results:"
        close = "Next step: use the relevant app section to generate any missing result before making a business decision."
    if facts:
        return lead + "\n\n" + "\n".join(f"- {fact}" for fact in facts[:5]) + "\n\n" + close
    return lead + "\n\nThe specific result needed for this question is not available yet.\n\n" + close
def _build_grounded_complete_answer(
    query: str,
    context: Dict[str, Any],
    intent_key: str = "",
    use_llm: bool = False,
    original_answer: str = "",
) -> str:
    if use_llm:
        return original_answer
    context = context or {}
    if _is_concept_query(query):
        return _handle_concept_question(query, context)
    if intent_key == "full_summary" or _is_full_summary_query(query):
        return _handle_full_summary(context)
    q = _normalize(query)
    if intent_key == "underfitting" or "underfit" in q:
        return _handle_underfitting(context)
    if intent_key == "overfitting" or "overfit" in q:
        return _handle_overfitting(context)
    if not _is_system_related_query(query):
        return _fallback_for_open_question(query, context)
    return _build_system_aware_answer(query, context)
def _enforce_answer_completeness(
    answer: str,
    query: str,
    context: Dict[str, Any],
    intent_key: str = "",
    use_llm: bool = False,
) -> str:
    if use_llm:
        return answer
    if _is_concept_query(query):
        return answer
    if intent_key == "full_summary" or _is_full_summary_query(query):
        return answer
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
    return _build_grounded_complete_answer(
        query,
        context,
        intent_key,
        use_llm=use_llm,
        original_answer=answer,
    )
def _handle_concept_question(query: str, context: Dict[str, Any]) -> str:
    q = _normalize(query)
    # Pure concept routes intentionally do not read session state. Mixed or
    # session-specific questions are handled by deterministic system routes.
    metrics: Dict[str, Any] = {}
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
        return "\n".join(lines)
    if "model" in q:
        lines = [
            "A machine learning model is a trained pattern-recognition system that learns from historical data and makes predictions on new records.",
            "",
            "Here, the model learns customer patterns linked to churn and outputs churn risk.",
        ]
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
def _sanitize_context_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _sanitize_context_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_context_value(v) for v in value]
    if isinstance(value, tuple):
        return [_sanitize_context_value(v) for v in value]
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return redact_text(str(value))
def redact_context(context: Dict[str, Any]) -> Dict[str, Any]:
    """Return redacted system context for the LLM without dropping useful app results."""
    if not context:
        return {}
    safe: Dict[str, Any] = {}
    for key in (
        "eda",
        "metrics",
        "predictions",
        "shap",
        "simulate",
        "segment",
        "segment_result",
        "lifecycle",
        "model_history",
        "available_models",
        "current_model",
    ):
        if key in context and context.get(key) is not None:
            safe[key] = _sanitize_context_value(context.get(key))
    if "segment" not in safe and "segment_result" in safe:
        safe["segment"] = safe["segment_result"]
    if "available_models" not in safe:
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
def _handle_model_advice_question(context: Dict[str, Any]) -> str:
    context = context or {}
    metrics = context.get("metrics") or {}
    history = context.get("model_history") or []
    current_model = _get_model_type(context)
    lines = [
        "For churn prediction, tree-based models like CatBoost, XGBoost, Random Forest, and Gradient Boosting are usually strong candidates.",
        "",
        "They work well because churn is often driven by non-linear interactions between tenure, contract type, usage, billing, and support behavior.",
    ]
    if current_model or metrics:
        lines.append("\nIn your system:")
        if current_model:
            lines.append(f"- Current model: {current_model}")
        metric_summary = _metric_summary_for_answer(context)
        if metric_summary:
            lines.append(f"- Current metrics: {metric_summary}")
        fit_status = metrics.get("fit_status") if isinstance(metrics, dict) else None
        if fit_status:
            lines.append(f"- Fit status: {fit_status}")
        recall = metrics.get("recall") if isinstance(metrics, dict) else None
        if recall is not None:
            lines.append(f"- Recall read: {_recall_verdict(recall)}")
    best_name = ""
    best_score = None
    if isinstance(history, list):
        for item in history:
            if not isinstance(item, dict) or item.get("suspicious"):
                continue
            m2 = item.get("metrics") or item
            score = _as_float(m2.get("roc_auc"))
            if score is None:
                score = _as_float(m2.get("f1"))
            if score is None:
                score = _as_float(m2.get("recall"))
            if score is None:
                continue
            if best_score is None or score > best_score:
                best_score = score
                best_name = str(item.get("model_type") or item.get("model") or "").strip()
    if best_name:
        lines.append(f"\nAmong your trained non-suspicious models, {best_name} currently looks strongest by the available comparison score.")
    else:
        lines.append("\nI would not declare a single best model from theory alone. Compare models on your validation/test data, then prioritize recall if the business goal is catching churners early.")
    lines.append("Practical next step: compare CatBoost/XGBoost/Random Forest, check train-test gap, and choose the model with the best recall-F1 balance rather than accuracy alone.")
    return "\n".join(lines)
def _handle_known_ml_question(query: str, context: Dict[str, Any]) -> str:
    q = _normalize(query)
    concept_markers = ("what is", "what does", "define", "explain", "meaning", "how does")
    if not any(marker in q for marker in concept_markers):
        return ""
    for model_name, explanation in MODEL_EXPLANATIONS.items():
        if model_name.lower() in q:
            lines = [
                f"{model_name} is a machine learning algorithm used for prediction tasks such as churn classification.",
                "",
                explanation,
            ]
            current_model = _get_model_type(context)
            if current_model and current_model.lower() == model_name.lower():
                lines.append(f"\nIn your system, {model_name} is the current trained model.")
            return "\n".join(lines)
    if "shap" in q or "feature importance" in q:
        lines = [
            "SHAP explains how much each feature contributes to a model's predictions.",
            "",
            "In churn prediction, SHAP helps turn model output into business drivers, such as tenure, contract type, or billing behavior.",
        ]
        features = _top_feature_names(context, top_n=3)
        if features:
            lines.append(f"\nIn your current system, the top available drivers are: {', '.join(features)}.")
        return "\n".join(lines)
    if "confusion matrix" in q:
        return (
            "A confusion matrix shows correct and incorrect predictions split into true positives, false positives, true negatives, and false negatives.\n\n"
            "For churn, false negatives are especially important because they are real churners the model missed."
        )
    if "threshold" in q:
        return (
            "A prediction threshold is the cutoff used to turn a churn probability into a yes/no churn prediction.\n\n"
            "Lowering the threshold usually increases recall, meaning the model catches more churners, but it can also create more false alarms."
        )
    return ""
def _general_missing_system_answer(query: str, context: Dict[str, Any] | None = None) -> str:
    q = _normalize(query)
    if "joke" in q:
        return "Why did the spreadsheet bring a ladder? Because it wanted to reach the next level of analysis."
    if any(k in q for k in ("eda", "exploratory", "dataset", "data summary")):
        return (
            "EDA means exploratory data analysis. In this webapp, it is the section that helps you understand the dataset before modeling: rows, columns, missing values, target column, distributions, and data quality.\n\n"
            "For your current session, I do not have a specific EDA result to quote, so I will not invent row counts or column names."
        )
    if "missing" in q or "null" in q:
        return (
            "Missing values are blank or unavailable entries in a dataset. They matter because models need a consistent input table.\n\n"
            "In this webapp, the EDA section reports missing-value counts, and preprocessing handles missing numeric values and categories before training. I do not have a current missing-value count to quote."
        )
    if any(k in q for k in ("reduce", "improve", "retain", "retention", "action", "recommend")):
        return (
            "To reduce churn, start by identifying at-risk customers, then target the reasons behind their risk: low engagement, contract friction, pricing concerns, poor onboarding, service issues, or weak loyalty signals.\n\n"
            "Once this webapp has model predictions and churn drivers, those actions can become much more specific."
        )
    if any(k in q for k in ("driver", "feature", "shap", "why", "reason", "cause")):
        return (
            "Churn drivers are the features that most influence churn risk. SHAP-style explanations help translate model behavior into business reasons, such as tenure, contract type, activity, geography, pricing, or usage patterns.\n\n"
            "I do not have current top-driver results here, so the safest answer is conceptual rather than data-specific."
        )
    if "churn rate" in q or "churn" in q:
        return (
            "Churn rate is the percentage of customers who leave, cancel, or stop using the service.\n\n"
            "For this webapp, actual churn comes from the training labels, while predicted churn comes from model predictions. I do not have both current values available here, so I will explain the concept without inventing numbers."
        )
    if any(k in q for k in ("metric", "accuracy", "recall", "precision", "f1", "auc", "roc")):
        return (
            "Model metrics explain how well the churn model performs. Accuracy measures overall correctness, recall measures how many real churners are caught, precision measures how clean the risk list is, F1 balances precision and recall, and ROC-AUC measures ranking quality.\n\n"
            "For churn decisions, recall is usually the most important metric because missed churners are missed retention opportunities."
        )
    if any(k in q for k in ("predict", "prediction", "risk", "high risk")):
        return (
            "Predictions are the model's estimate of which customers are likely to churn. In this webapp, prediction results can include predicted churn rate, predicted churn count, risk tiers, high-risk customers, and revenue at risk.\n\n"
            "I do not have a current prediction result to quote, so I will not make up customer counts."
        )
    if any(k in q for k in ("model", "algorithm", "train", "training")):
        return (
            "For churn prediction, models like Logistic Regression, Random Forest, XGBoost, Gradient Boosting, and CatBoost are common choices. The best one depends on validation performance, recall, F1, ROC-AUC, and whether it generalizes well.\n\n"
            "I do not have current trained-model results to compare, so I can explain the options generally but will not claim which model won in your session."
        )
    return (
        "I can answer generally. If the question needs a specific number from this churn webapp, that result is not available in the current context, so I will not invent it."
    )
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
        return _general_missing_system_answer("churn rate", context)
    lines: List[str] = []
    if actual_rate is not None:
        level = "high" if actual_rate > 0.4 else "moderate" if actual_rate > 0.2 else "relatively low"
        lines.append(f"**Actual churn rate: {_fmt_pct(actual_rate, 2)}**")
        lines.append(f"This is the ground-truth churn rate from your training labels, and it is {level}.")
    else:
        lines.append("**Actual churn rate: not available yet**")
        lines.append("This specific value comes from the selected churn label in training results.")
    if predicted_rate is not None:
        lines.append(f"**Predicted churn rate: {_fmt_pct(predicted_rate, 2)}**")
    else:
        lines.append("**Predicted churn rate: not available yet**")
        lines.append("This specific value comes from the prediction results section.")
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
        return _general_missing_system_answer("churn drivers", context)
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
        return _general_missing_system_answer("model metrics", context)
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
        lines.append("Top churn drivers are not available in the current results, so I can explain the metrics but not the specific causes behind them.")
    lines.append("Next step: optimize for recall if the business goal is to catch more at-risk customers before they leave.")
    return "\n".join(lines)
def _handle_simulation(context: Dict[str, Any]) -> str:
    sim = context.get("simulate")
    if not sim:
        return (
            "A retention simulation estimates whether an intervention is worth the cost. It usually compares expected churn before and after an action, retained customers, revenue saved, campaign cost, and ROI.\n\n"
            "I do not have a current simulation result to quote, so I can explain the idea but will not invent ROI or revenue saved."
        )
    return "Here's a summary of the last retention simulation:\n\n" + _format_simulation(sim)
def _handle_columns(context: Dict[str, Any]) -> str:
    eda = context.get("eda") or {}
    cols = eda.get("columns")
    if not cols:
        return _general_missing_system_answer("dataset columns", context)
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
            "To reduce churn generally, focus on at-risk customers, improve onboarding, address pricing or contract friction, increase engagement, and personalize retention offers.\n\n"
            "When model drivers are available, I can turn those general actions into dataset-specific recommendations."
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
        return _general_missing_system_answer("eda dataset", context)
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
        return _general_missing_system_answer("missing values", context)
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
        return _general_missing_system_answer("model training", context)
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
        "Segment analysis compares churn across groups, such as geography, contract type, lifecycle stage, tenure band, or activity level.\n\n"
        "It is useful because churn is rarely uniform: one segment may need onboarding help while another needs pricing or loyalty action. I do not have a current segment breakdown to quote here."
    )
def _handle_predictions_summary(context: Dict[str, Any]) -> str:
    predictions = context.get("predictions") or {}
    if not predictions:
        return _general_missing_system_answer("predictions", context)
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
            "High-risk customers are the customers whose predicted churn probability crosses the risk threshold, often used as the first retention target list.\n\n"
            "I do not have a current high-risk customer count to quote, so I can explain the idea but will not invent a number."
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
        return _general_missing_system_answer("dataset statistics", context)
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
            "Model training is the step where the webapp learns patterns that separate churners from non-churners.\n\n"
            "In this system, training usually means choosing a churn target column, selecting an algorithm such as Logistic Regression, Random Forest, XGBoost, or CatBoost, then evaluating metrics like recall, F1, ROC-AUC, and fit status. I do not have current training results to quote."
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
            "A full churn summary normally combines dataset quality, model metrics, actual vs predicted churn, prediction risk tiers, top churn drivers, and retention recommendations.\n\n"
            "I do not have current section results to quote, so I can describe the summary structure but will not invent values."
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
        lines.append("- Top churn drivers are not available in the current results")
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
            "Model tracking compares trained algorithms so you can choose the model that best balances recall, precision, F1, ROC-AUC, and generalization.\n\n"
            "I do not have current model-history results to quote, so I can explain how comparison works but will not name a winning model."
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
            "Different churn models have different strengths. Logistic Regression is interpretable, Random Forest handles non-linear patterns, XGBoost and CatBoost are strong for tabular churn data, and Naive Bayes is fast and simple.\n\n"
            "I do not have a current trained model to explain, so I can compare the algorithms generally but will not claim which one your session used."
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
            "Overfitting means a model learns the training data too closely and performs worse on new data.\n\n"
            "To confirm it for this webapp, I would need train/test metrics or fit status. Those values are not available in the current context, so I can explain the concept but will not diagnose your model."
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
def _handle_underfitting(context: Dict[str, Any]) -> str:
    metrics = context.get("metrics") or {}
    if not metrics:
        return (
            "Underfitting cannot be checked yet because no model metrics are available.\n\n"
            "Underfitting means the model is too simple and fails to learn real patterns. "
            "It usually performs poorly on both training data and new data."
        )
    fit_status = metrics.get("fit_status") or metrics.get("fit") or ""
    train_score = _as_float(metrics.get("train_score") or metrics.get("train_f1"))
    test_score = _as_float(metrics.get("test_score") or metrics.get("f1"))
    lines = [
        "Underfitting means the model is too simple and fails to learn real patterns.",
        "",
        "It usually performs poorly on both training data and new data.",
    ]
    model_type = _get_model_type(context)
    if model_type:
        lines.append(f"\nCurrent model: {model_type}.")
    if train_score is not None and test_score is not None:
        train_rate = _as_rate(train_score)
        test_rate = _as_rate(test_score)
        if train_rate is not None and test_rate is not None:
            if train_rate < 0.60 and test_rate < 0.60:
                verdict = "This does look like underfitting because both train and test scores are weak."
            elif abs(train_rate - test_rate) <= 0.03 and test_rate < 0.70:
                verdict = "This may be mild underfitting because the gap is small but performance is still only moderate."
            else:
                verdict = "This does not look like classic underfitting from the available train/test scores."
            lines.append(
                f"Train score is {_fmt_pct(train_rate)} and test score is {_fmt_pct(test_rate)}. {verdict}"
            )
    elif fit_status:
        lines.append(f"Your system's fit status is **{fit_status}**.")
    else:
        lines.append("Train/test comparison is not available, so underfitting cannot be confirmed from the current context.")
    lines.append("\nNext step: if underfitting is present, try stronger features, a more expressive model, or better preprocessing before tuning thresholds.")
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
    "underfitting": _handle_underfitting,
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
    (re.compile(r"\b(help|what can you do|commands|how do i use (this|the app|the system)|how to use (this|the app|the system))\b"), "help"),
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
    (re.compile(r"\b(underfit|underfitting)\b"), "underfitting"),
    (re.compile(r"\b(overfit|overfitting|generaliz(e|ation)|train.?test gap|fit status)\b"), "overfitting"),
    (re.compile(r"\b(training (detail|info|process)|how (was|is) (the )?model trained|training mode|target column|which column)\b"), "training_details"),
    # model tracking & comparison — "which model" removed to avoid conflict with available_models
    (re.compile(r"\b(compare (my |our |trained |current )?models?|best (trained|current|my|our) model|model comparison|models? trained)\b"), "model_tracking"),
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
    "underfitting": ["is my model underfitting", "underfit", "underfitting"],
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
def _build_user_prompt(query: str, safe_context: Dict[str, Any]) -> str:
    context_json = json.dumps(safe_context, indent=2, sort_keys=True)
    return (
        "User Question:\n"
        f"{redact_text(query)}\n\n"
        "System Data (JSON):\n"
        f"{context_json}\n\n"
        "Instructions:\n"
        "- Use system data ONLY if relevant.\n"
        "- If question is general, ignore system data.\n"
        "- If question is about the model, predictions, churn, segments, simulation, or data, use system data.\n"
        "- Never invent, modify, or assume numeric values."
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
def _answer_with_llm(query: str, context: Dict[str, Any], llm_provider: str, llm_model: str) -> str:
    if llm_provider.lower() != "openai":
        raise RuntimeError("That LLM provider is not supported yet. Please use the default OpenAI option.")
    safe_ctx = redact_context(context or {})
    system_prompt = (
        "You are a smart AI assistant for a churn analysis system.\n\n"
        "You behave like ChatGPT:\n"
        "- Answer naturally and clearly.\n"
        "- Answer ANY question, including random ones.\n\n"
        "CRITICAL RULES:\n"
        "- If system data is provided, treat it as TRUE.\n"
        "- NEVER change or invent numbers.\n"
        "- NEVER assume missing values.\n"
        "- Use system data ONLY when relevant.\n\n"
        "RESPONSE STYLE:\n"
        "- No forced bullet dumps.\n"
        "- No robotic structure.\n"
        "- Explain like a human."
    )
    user_prompt = _build_user_prompt(query, safe_ctx)
    return redact_text(_call_openai_chat(system_prompt, user_prompt, model=llm_model))
def _llm_failure_notice(error: Exception) -> str:
    err_str = str(error)
    if "429" in err_str or "insufficient_quota" in err_str or "quota" in err_str.lower():
        return (
            "*(AI assistant unavailable - API quota reached. "
            "Showing the best grounded system answer below.)*\n\n"
        )
    if "auth" in err_str.lower() or "api_key" in err_str.lower() or "401" in err_str:
        return (
            "*(AI assistant unavailable - API key issue. "
            "Showing the best grounded system answer below.)*\n\n"
        )
    return (
        "*(AI assistant encountered a temporary issue - "
        "showing the best grounded system answer below.)*\n\n"
    )
# -------------------------
# Fallback helper for open questions
# -------------------------
def _fallback_for_open_question(query: str, context: Dict[str, Any]) -> str:
    q = (query or "").lower()
    if _is_concept_query(query):
        return _handle_concept_question(query, context or {})
    if _is_full_summary_query(query):
        return _handle_full_summary(context or {})
    if _is_model_advice_query(query):
        return _handle_model_advice_question(context or {})
    known_answer = _handle_known_ml_question(query, context or {})
    if known_answer:
        return known_answer
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
    if "underfit" in q:
        return _handle_underfitting(context or {})
    if "overfit" in q or "train test" in q or "fit status" in q:
        return _handle_overfitting(context or {})
    if "model" in q or "train" in q or "algorithm" in q:
        return _handle_model_info(context or {})
    if _is_system_related_query(query):
        return _build_system_aware_answer(query, context or {})
    return _general_missing_system_answer(query, context)
# -------------------------
# Public API
# -------------------------
def respond_to_query(query: str, context: Dict[str, Any], use_llm: bool = True, llm_provider: str = "openai", llm_model: str = "gpt-4o-mini") -> str:
    """
    Main chatbot entrypoint.

    LLM mode is the primary architecture: when enabled, call the LLM and return
    immediately. Deterministic handlers are only for non-LLM fallback mode.
    """
    if not query or not isinstance(query, str):
        return "It looks like your message was empty. Try asking something like 'What is the churn rate?' or 'Show me the top churn factors.'"
    context = context or {}

    # LLM mode is final. No intent matching, handlers, enforcement, grounded
    # builders, or fallback logic may run after this return.
    if use_llm:
        return _answer_with_llm(query, context, llm_provider, llm_model)

    if (re.search(r"[\w\.-]+@[\w\.-]+\.\w+", query)
            or re.search(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b", query)
            or re.search(r"\b\d{10}\b", query)):
        return (
            "For privacy reasons, I can't process personal identifiers like email addresses, "
            "phone numbers, or ID numbers in chat.\n\n"
            "To look up a specific customer, use the **Explain** tab and enter their row number from the dataset."
        )
    q_low = _normalize(query)

    def safe_handler(intent_key: str, score: float | None = None) -> str:
        try:
            return _answer_from_handler(intent_key, query, context, score)
        except Exception:
            return "Something went wrong while generating that answer. Try rephrasing your question."

    # 1. Strict system routes are deterministic because they answer exact app
    # facts or app commands.
    intent_key, _score = _match_intent(query)
    if intent_key in _PRE_LLM_STRICT_INTENTS:
        return safe_handler(intent_key)

    # 2. Pure concept questions must stay clean: no churn state, no metrics,
    # no "train a model first" answer unless the user asks about this session.
    if _is_concept_query(query):
        return _handle_concept_question(query, context)

    llm_notice = ""

    # 4. Offline/fallback deterministic routes.
    if _is_general_chat_query(query):
        return llm_notice + (
            "I am the assistant inside this churn analysis app. I can explain ML ideas, "
            "help interpret churn results, and answer normal questions. When the AI mode is on, "
            "I can handle broader conversation more flexibly; when it is off, I stick to a smaller built-in answer set."
        )

    if _is_model_advice_query(query):
        return llm_notice + _handle_model_advice_question(context)

    if "predict" in q_low and any(k in q_low for k in ("actual", "lower", "higher", "underestimate", "overestimate", "different")):
        return llm_notice + _build_system_aware_answer(query, context)

    # 5. Exact data questions use deterministic handlers only when LLM is off
    # or unavailable.
    if intent_key in _STRICT_HANDLER_INTENTS:
        return llm_notice + safe_handler(intent_key)

    # 6. Hybrid/system analysis uses the specific grounded handler.
    if intent_key in _GROUNDED_HANDLER_INTENTS:
        return llm_notice + safe_handler(intent_key)

    # 7. Broad questions about this system get a grounded synthesized read.
    if _needs_completeness_guard(query):
        return llm_notice + _build_grounded_complete_answer(query, context, use_llm=False)

    # 8. Fuzzy matching only rescues likely system questions, not general chat.
    system_markers = (
        "churn", "customer", "model", "metric", "predict", "prediction",
        "dataset", "data", "row", "column", "feature", "risk", "segment",
        "lifecycle", "simulation", "roi", "train", "accuracy", "recall",
        "precision", "f1", "auc",
    )
    if any(marker in q_low for marker in system_markers):
        fuzzy_intent, fuzzy_score = _match_intent(query, fuzzy_threshold=0.6)
        if fuzzy_intent and fuzzy_intent != intent_key:
            return llm_notice + safe_handler(fuzzy_intent, fuzzy_score)

    return llm_notice + _fallback_for_open_question(query, context)

