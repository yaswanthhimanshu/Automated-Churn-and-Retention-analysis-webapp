"""
chatbot.py

Hybrid chatbot for ChurnInsight.

Public API:
    respond_to_query(query: str, context: dict, use_llm: bool = False, llm_provider: str = "openai", llm_model: str = "gpt-4o-mini") -> str

Notes:
- context is a dict containing keys like: "eda", "metrics", "shap", "simulate"
- This module NEVER sends raw dataframe rows to an LLM. It builds a redacted summary.
- Optional dependencies: openai, scikit-learn. The code works without them (LLM/fuzzy TF-IDF disabled).
"""

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
        return "No SHAP feature summary available."
    lines = []
    for f in features:
        name = f.get("name", "<unknown>") if isinstance(f, dict) else str(f)
        val = f.get("mean_abs_shap") if isinstance(f, dict) else None
        if val is not None:
            try:
                lines.append(f"- {name} (mean |SHAP| = {float(val):.4f})")
            except Exception:
                lines.append(f"- {name} (mean |SHAP| = {val})")
        else:
            lines.append(f"- {name}")
    return "\n".join(lines)

def _format_metrics(metrics: Dict[str, Any]) -> str:
    if not metrics:
        return "No model metrics available."
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
        return "No simulation results available."
    lines = []
    before = sim.get("before_churn_rate")
    after = sim.get("after_churn_rate")
    retained = sim.get("retained_customers")
    revenue = sim.get("revenue_saved")
    cost = sim.get("action_cost")
    roi = sim.get("roi")
    if before is not None:
        try:
            lines.append(f"Before churn rate: {float(before):.3%}")
        except Exception:
            lines.append(f"Before churn rate: {before}")
    if after is not None:
        try:
            lines.append(f"After churn rate: {float(after):.3%}")
        except Exception:
            lines.append(f"After churn rate: {after}")
    if retained is not None:
        try:
            lines.append(f"Estimated retained customers: {int(retained)}")
        except Exception:
            lines.append(f"Estimated retained customers: {retained}")
    if revenue is not None:
        try:
            lines.append(f"Estimated revenue saved: ${float(revenue):,.2f}")
        except Exception:
            lines.append(f"Estimated revenue saved: {revenue}")
    if cost is not None:
        try:
            lines.append(f"Action cost: ${float(cost):,.2f}")
        except Exception:
            lines.append(f"Action cost: {cost}")
    if roi is not None:
        try:
            lines.append(f"ROI: {float(roi):.2f}%")
        except Exception:
            lines.append(f"ROI: {roi}")
    return "\n".join(lines) if lines else "Simulation result contained no recognizable fields."

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
    return safe

# -------------------------
# Rule-based intent handlers
# -------------------------
def _handle_help(_context: Dict[str, Any]) -> str:
    return (
        "I can answer questions about churn, model performance, feature importance (SHAP), "
        "and retention simulations. Examples:\n"
        "- 'What is the churn rate?'\n"
        "- 'Why are customers churning?'\n"
        "- 'What is the ROI of the last simulation?'\n"
        "Use the Explain page for per-customer details."
    )

def _handle_churn_rate(context: Dict[str, Any]) -> str:
    sim = context.get("simulate")
    if sim and "before_churn_rate" in sim:
        try:
            return f"Baseline churn rate: {float(sim['before_churn_rate']):.3%}"
        except Exception:
            return f"Baseline churn rate: {sim['before_churn_rate']}"
    eda = context.get("eda") or {}
    for key in ("churn_rate", "target_churn_rate", "target_rate"):
        if key in eda:
            try:
                return f"Churn rate (from EDA): {float(eda[key]):.3%}"
            except Exception:
                return f"Churn rate: {eda[key]}"
    metrics = context.get("metrics") or {}
    if "churn_rate" in metrics:
        return f"Churn rate (from metrics): {metrics['churn_rate']}"
    return "Churn rate not found — run EDA or a simulation."

def _handle_top_drivers(context: Dict[str, Any]) -> str:
    shap = context.get("shap")
    features = _shortlist_top_features_from_shap(shap, top_n=5)
    if not features:
        return "No SHAP summary available. Train a model and compute SHAP first."
    return "Top drivers of churn (by mean |SHAP|):\n" + _format_feature_list(features)

def _handle_metrics(context: Dict[str, Any]) -> str:
    m = context.get("metrics")
    if not m:
        return "No model metrics available. Train a model first."
    return "Model performance: " + _format_metrics(m)

def _handle_simulation(context: Dict[str, Any]) -> str:
    sim = context.get("simulate")
    if not sim:
        return "No simulation results available. Run a simulation to estimate ROI."
    return _format_simulation(sim)

def _handle_columns(context: Dict[str, Any]) -> str:
    eda = context.get("eda") or {}
    cols = eda.get("columns")
    if not cols:
        return "No column list available. Upload a dataset to view columns."
    if isinstance(cols, list):
        names = [c["name"] if isinstance(c, dict) and "name" in c else str(c) for c in cols]
        return f"Columns ({len(names)}): " + ", ".join(names[:50]) + ("..." if len(names) > 50 else "")
    return "Column list present but not in expected format."

def _handle_explain_customer(_context: Dict[str, Any]) -> str:
    return ("For privacy, I do not fetch raw customer rows in chat. Use the Explain page and provide a row index "
            "or upload a single-customer CSV to get a detailed SHAP explanation.")

def _handle_recommend_actions(context: Dict[str, Any]) -> str:
    shap = context.get("shap") or {}
    top = (shap.get("top_features") if isinstance(shap, dict) else None) or []
    sim = context.get("simulate")
    lines = []
    if top:
        lines.append("Based on the top SHAP features, consider these prioritized actions:")
        for f in top[:3]:
            name = f.get("name") if isinstance(f, dict) else str(f)
            lname = (name or "").lower()
            if "tenure" in lname:
                action = "Improve onboarding and early engagement for new users (welcome flows, tutorials)."
            elif "monthly" in lname or "price" in lname or "charges" in lname:
                action = "Target high-monthly-charge customers with tailored offers or value plans."
            elif "contract" in lname:
                action = "Offer incentives to convert month-to-month customers to longer contracts."
            elif "support" in lname or "ticket" in lname:
                action = "Provide proactive support and SLA for high-touch customers."
            elif "usage" in lname or "feature" in lname:
                action = "Increase engagement via in-app nudges and feature walkthroughs."
            else:
                action = "Investigate and design targeted interventions for this feature."
            lines.append(f"- {name}: {action}")
    else:
        lines.append("I don't have SHAP insights available. Train a model and compute SHAP to get data-driven suggestions.")

    if sim:
        lines.append("\nLast simulation summary (related actions):")
        try:
            if "roi" in sim:
                lines.append(f"- ROI: {float(sim['roi']):.2f}%")
        except Exception:
            pass
        try:
            if "retained_customers" in sim:
                lines.append(f"- Estimated retained customers: {int(sim['retained_customers'])}")
        except Exception:
            pass

    lines.append("\nI can run a simulation for any suggested action on request (to estimate retained customers and ROI).")
    return "\n".join(lines)

_INTENT_HANDLERS = {
    "help": _handle_help,
    "churn_rate": _handle_churn_rate,
    "top_drivers": _handle_top_drivers,
    "metrics": _handle_metrics,
    "simulation": _handle_simulation,
    "columns": _handle_columns,
    "explain_customer": _handle_explain_customer,
    "recommend_actions": _handle_recommend_actions
}

# -------------------------
# Intent patterns & examples
# -------------------------
_INTENTS = [
    (re.compile(r"\b(help|what can you do|how to|commands)\b"), "help"),
    (re.compile(r"\b(churn rate|what is the churn|current churn|how many left|customers left|how many churn)\b"), "churn_rate"),
    (re.compile(r"\b(why (are )?customers churning|drivers of churn|reasons for churn|why people leave)\b"), "top_drivers"),
    (re.compile(r"\b(top features|feature importance|top 5 features|top drivers)\b"), "top_drivers"),
    (re.compile(r"\b(model metrics|how good|accuracy|precision|recall|f1|roc_auc|roc auc)\b"), "metrics"),
    (re.compile(r"\b(simulation|roi|return on investment|revenue saved|action cost|how much do we save)\b"), "simulation"),
    (re.compile(r"\b(explain customer|explain user|explain row|explain id)\b"), "explain_customer"),
    (re.compile(r"\b(columns|what columns|list columns|schema|fields)\b"), "columns"),
    (re.compile(r"\b(what should i do|recommend|how to improve|improve sales|improve retention)\b"), "recommend_actions"),
]

_INTENT_EXAMPLES = {
    "help": ["what can you do", "help me", "how to use"],
    "churn_rate": ["what is the churn rate", "how many customers churn", "how many left"],
    "top_drivers": ["why are customers churning", "drivers of churn", "what causes churn"],
    "metrics": ["show model metrics", "what is the accuracy", "what is f1 score"],
    "simulation": ["what is the roi", "how much revenue saved", "simulate retention action"],
    "explain_customer": ["explain customer 123", "why did this customer churn"],
    "columns": ["list columns", "what columns are in the dataset"],
    "recommend_actions": ["what should i do to improve retention", "recommend actions to improve sales"]
}

_SYNONYMS = {
    "left": ["churned", "cancelled", "left", "resigned"],
    "roi": ["return on investment", "return", "roi%"],
    "revenue": ["income", "sales", "revenue_saved", "saved"],
    "accuracy": ["acc", "precision", "recall", "f1"],
    "customer": ["user", "client", "subscriber"],
    "columns": ["schema", "fields", "features"],
    "simulate": ["simulation", "simulate", "what if"]
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
        "You are a concise, factual assistant specialized in customer churn analysis. "
        "Answer only using the provided 'Context' below. Do not invent numbers. "
        "If data is missing, say so and suggest next steps (train model, compute SHAP, run simulation)."
    )

def _build_user_prompt(query: str, safe_context: Dict[str, Any]) -> str:
    parts = []
    parts.append("User question:")
    parts.append(redact_text(query))
    parts.append("\nContext summary (redacted):")
    if not safe_context:
        parts.append("No analysis context provided.")
    else:
        eda = safe_context.get("eda")
        if eda:
            parts.append(f"- EDA: shape={eda.get('shape') or eda.get('n_rows')}, cols={eda.get('n_cols')}")
            cols = eda.get("columns")
            if cols:
                parts.append(f"- Columns: {', '.join(cols[:30])}" + ("..." if len(cols) > 30 else ""))
        metrics = safe_context.get("metrics")
        if metrics:
            parts.append("- Model metrics: " + ", ".join([f"{k}={v}" for k, v in metrics.items()]))
        shap = safe_context.get("shap")
        if shap:
            top = shap.get("top_features", [])[:10]
            if top:
                parts.append("- Top SHAP features:")
                for f in top:
                    parts.append(f"  - {f.get('name')}: mean_abs_shap={f.get('mean_abs_shap')}")
        sim = safe_context.get("simulate")
        if sim:
            parts.append("- Last simulation summary:")
            for k, v in sim.items():
                parts.append(f"  - {k}: {v}")
    parts.append("\nInstructions:")
    parts.append("Answer in 3-6 short bullets. When suggesting an action, include which cohort to target and a next step (e.g. run simulation).")
    return "\n".join(parts)

def _call_openai_chat(system_prompt: str, user_prompt: str, model: str, max_tokens: int = 800, temperature: float = 0.0, timeout: int = 20) -> str:
    if not OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI SDK not available. Install `openai` to enable LLM mode.")
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")
    
    # ✅ Initialize the new OpenAI client
    client = OpenAI(api_key=api_key)
    
    # ✅ Use the new Chat Completions endpoint
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    # ✅ Return the chat content
    if resp and resp.choices and len(resp.choices) > 0:
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
    return ("I couldn't answer that precisely with the current analysis. Try: 'What is the churn rate?', "
            "'Why are customers churning?', or train a model and compute SHAP to get more detailed recommendations.")

# -------------------------
# Public API
# -------------------------
def respond_to_query(query: str, context: Dict[str, Any], use_llm: bool = False, llm_provider: str = "openai", llm_model: str = "gpt-4o-mini") -> str:
    """
    Main entrypoint for the chatbot.
    - query: the user text
    - context: dict with keys 'eda', 'metrics', 'shap', 'simulate'
    - use_llm: explicit opt-in flag (frontend must set)
    """
    if not query or not isinstance(query, str):
        return "Empty query. Ask about churn, metrics, feature importance, or simulations."

    # quick PII guard
    if re.search(r"[\w\.-]+@[\w\.-]+\.\w+", query) or re.search(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b", query) or re.search(r"\b\d{10}\b", query):
        return "I cannot process direct personal identifiers (email, phone, SSN) in chat. Use the Explain page."

    q_low = _normalize(query)

    # 1) precise regex intent matching
    for pattern, intent_key in _INTENTS:
        if pattern.search(q_low):
            handler = _INTENT_HANDLERS.get(intent_key)
            if handler:
                try:
                    return handler(context or {})
                except Exception:
                    return "Handler error; try a simpler question."

    # 2) fuzzy matching (difflib + optional TF-IDF)
    intent_key, score = _fuzzy_intent_match(query)
    if intent_key and score >= 0.6:
        handler = _INTENT_HANDLERS.get(intent_key)
        if handler:
            try:
                resp = handler(context or {})
                return resp + f"\n\n(interpreted intent: {intent_key}, confidence: {score:.2f})"
            except Exception:
                pass

    # 3) if LLM enabled & allowed, call it (on redacted context)
    if use_llm:
        try:
            safe_ctx = redact_context(context or {})
            system_prompt = _build_system_prompt()
            user_prompt = _build_user_prompt(query, safe_ctx)
            if llm_provider.lower() == "openai":
                answer = _call_openai_chat(system_prompt, user_prompt, model=llm_model)
                return redact_text(answer)
            else:
                return "LLM provider not supported."
        except Exception as e:
            try:
                fallback = _fallback_for_open_question(query, context or {})
                return f"(LLM failed: {str(e)})\n\nFallback:\n{fallback}"
            except Exception:
                return f"LLM failed and fallback unavailable: {str(e)}"

    # 4) Final deterministic fallback
    if "recommend" in q_low or "what should i do" in q_low or "improve" in q_low:
        return _handle_recommend_actions(context or {})

    return (
        "I don't fully understand that. Try one of these:\n"
        "- 'What is the churn rate?'\n"
        "- 'Why are customers churning?'\n"
        "- 'Show top features driving churn.'\n"
        "- 'What was the ROI of the last simulation?'\n"
        "Or enable the LLM assistant for broader recommendations (requires opt-in)."
    )
