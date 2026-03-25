from typing import List, Dict, Any, Optional, Union
import pandas as pd


_PERIOD_KEYWORDS = [
    "month", "monthly",
    "year", "yearly", "annual", "annually",
    "quarter", "quarterly",
    "week", "weekly",
    "daily"
]

_CHURN_POSITIVE_VALUES = {
    "1", "yes", "true", "y",
    "churn", "churned", "positive"
}


def _looks_like_id(col: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(col):
        return False
    if col.isna().all():
        return False
    return (col.nunique() / len(col)) > 0.9


def _contains_time_words(col: pd.Series) -> bool:
    for v in col.dropna().astype(str).unique():
        v = v.lower()
        for kw in _PERIOD_KEYWORDS:
            if kw in v:
                return True
    return False


def _is_churn_positive(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float) > 0
    return (
        series.astype(str)
        .str.lower()
        .str.strip()
        .isin(_CHURN_POSITIVE_VALUES)
    )


def detect_time_candidates(df: pd.DataFrame) -> List[Dict[str, Any]]:
    candidates = []

    for col_name in df.columns:
        col = df[col_name]

        if pd.api.types.is_numeric_dtype(col):
            if _looks_like_id(col):
                continue

            non_null = col.dropna()
            if non_null.empty:
                continue

            min_v = non_null.min()
            max_v = non_null.max()
            unique_vals = non_null.nunique()

            if min_v >= 0 and unique_vals > 5:
                confidence = 0.6

                if max_v <= 120:
                    confidence += 0.3
                elif max_v <= 365:
                    confidence += 0.15
                else:
                    confidence -= 0.2

                confidence = min(max(confidence, 0.0), 0.95)

                candidates.append({
                    "column": col_name,
                    "type": "duration",
                    "confidence": round(confidence, 2),
                    "reason": f"numeric non-negative values (min={min_v}, max={max_v})"
                })

        elif pd.api.types.is_object_dtype(col) or pd.api.types.is_categorical_dtype(col):
            non_null = col.dropna()
            if non_null.empty:
                continue

            unique_vals = non_null.nunique()
            if unique_vals > 20:
                continue

            if _contains_time_words(non_null):
                confidence = 0.75
                if unique_vals <= 5:
                    confidence += 0.15

                candidates.append({
                    "column": col_name,
                    "type": "period",
                    "confidence": round(min(confidence, 0.95), 2),
                    "reason": "categorical values contain time-related words"
                })

    candidates.sort(key=lambda x: x["confidence"], reverse=True)
    return candidates


def filter_by_time(
    df: pd.DataFrame,
    time_column: str,
    time_type: str,
    user_value: Union[int, float, str, list, tuple]
) -> pd.DataFrame:

    if time_column not in df.columns:
        raise ValueError(f"{time_column} not found")

    if time_type == "duration":

        if isinstance(user_value, (list, tuple)) and len(user_value) == 2:
            min_v, max_v = user_value
            return df[(df[time_column] >= min_v) & (df[time_column] <= max_v)]

        return df[df[time_column] <= user_value]

    if time_type == "period":

        if not isinstance(user_value, (list, tuple)):
            user_value = [user_value]

        return df[df[time_column].astype(str).isin([str(v) for v in user_value])]

    raise ValueError("time_type must be 'duration' or 'period'")


def summarize_actual_churn(
    df: pd.DataFrame,
    target_col: str
) -> Dict[str, Any]:

    if target_col not in df.columns:
        raise ValueError("Target column not found")

    series = df[target_col].dropna()
    total = len(series)

    if total == 0:
        return {
            "customers": 0,
            "churned": 0,
            "not_churned": 0,
            "churn_rate": 0.0
        }

    churn_mask = _is_churn_positive(series)
    churned = int(churn_mask.sum())
    not_churned = total - churned

    return {
        "customers": total,
        "churned": churned,
        "not_churned": not_churned,
        "churn_rate": round(churned / total, 4)
    }



def summarize_predicted_churn(
    df: pd.DataFrame,
    pred_col: str = "predicted_churn",
    prob_col: str = "churn_probability"
) -> Optional[Dict[str, Any]]:

    if pred_col not in df.columns:
        return None

    total = len(df)

    preds = df[pred_col]
    churned = int((preds.astype(float) > 0).sum())

    avg_prob = (
        float(df[prob_col].mean())
        if prob_col in df.columns else None
    )

    high_risk = (
        int((df[prob_col] >= 0.7).sum())
        if prob_col in df.columns else None
    )

    return {
        "customers": total,
        "predicted_churn": churned,
        "high_risk": high_risk,
        "avg_churn_probability": round(avg_prob, 4) if avg_prob is not None else None
    }

def get_column_unique_values(
    df: pd.DataFrame,
    column: str
) -> List[Dict[str, Any]]:

    if column not in df.columns:
        raise ValueError(f"{column} not found")

    counts = df[column].value_counts(dropna=False)

    return [
        {
            "value": str(val),
            "count": int(cnt)
        }
        for val, cnt in counts.items()
    ]

def groupby_churn_analysis(
    df: pd.DataFrame,
    column: str,
    target_col: str
) -> Dict[str, Any]:

    if column not in df.columns:
        raise ValueError(f"{column} not found")

    if target_col not in df.columns:
        raise ValueError("Target column not found")

    result = []

    grouped = df.dropna(subset=[column]).groupby(column)

    for value, group in grouped:
        stats = summarize_actual_churn(group, target_col)

        result.append({
            "value": str(value) if pd.notna(value) else "Unknown",
            "customers": stats["customers"],
            "churned": stats["churned"],
            "not_churned": stats["not_churned"],
            "churn_rate": stats["churn_rate"]
        })

    # sort by highest churn
    result = sorted(result, key=lambda x: x["churn_rate"], reverse=True)

    return {
        "analysis_type": "categorical",
        "column": column,
        "segments": result
    }


def analyze_time_churn(
    df: pd.DataFrame,
    target_col: str,
    time_column: str,
    time_type: str,
    user_value,
    predictions_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:

    
    if not pd.api.types.is_numeric_dtype(df[time_column]):

        group_result = groupby_churn_analysis(
            df=df,
            column=time_column,
            target_col=target_col
        )

        return {
            "mode": "groupby",
            "column": time_column,
            "result": group_result
        }

    
    filtered_df = filter_by_time(
        df=df,
        time_column=time_column,
        time_type=time_type,
        user_value=user_value
    )

    result = {
        "mode": "filtered",
        "time_filter": {
            "column": time_column,
            "type": time_type,
            "value": user_value,
            "customers_after_filter": len(filtered_df)
        },
        "observed_churn": summarize_actual_churn(
            filtered_df, target_col
        )
    }

    if predictions_df is not None and time_column in predictions_df.columns:
        filtered_pred = filter_by_time(
            df=predictions_df,
            time_column=time_column,
            time_type=time_type,
            user_value=user_value
        )
        result["predicted_churn"] = summarize_predicted_churn(filtered_pred)

    return result


# Keywords that indicate a column is a valid lifecycle/tenure duration
_LIFECYCLE_ALLOW_KEYWORDS = {"tenure", "month", "months", "duration", "time"}

# Keywords that disqualify a column even if it's numeric and non-negative
_LIFECYCLE_REJECT_KEYWORDS = {"age", "id", "code", "salary", "score", "zip", "pin", "year"}


def _is_valid_lifecycle_column(col_name: str) -> bool:
    """
    Return True only if the column name contains an allowed lifecycle keyword
    AND does not contain a disqualifying keyword.
    Matching is case-insensitive and checks whole-word substrings.
    """
    name_lower = col_name.lower()
    # Reject immediately if a disqualifying word appears in the column name
    for reject in _LIFECYCLE_REJECT_KEYWORDS:
        if reject in name_lower:
            return False
    # Accept only if at least one lifecycle keyword is present
    for allow in _LIFECYCLE_ALLOW_KEYWORDS:
        if allow in name_lower:
            return True
    return False


def lifecycle_risk_analysis(
    df: pd.DataFrame,
    target_col: str,
    predictions_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Analyse churn across customer lifecycle stages (Early ≤3 , Mid 4-12 , Late >12).

    Only operates on columns whose name clearly indicates tenure/months duration.
    Returns a dict that always has an 'error' key when no valid column is found,
    so callers and the frontend can surface a clear message instead of wrong results.
    """

    # --- Step 1: find a numeric column whose name signals lifecycle meaning ---
    lifecycle_col = None
    for col_name in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col_name]):
            continue
        if _looks_like_id(df[col_name]):
            continue
        if _is_valid_lifecycle_column(col_name):
            lifecycle_col = col_name
            break  # take the first valid match

    if lifecycle_col is None:
        return {
            "error": (
                "No valid lifecycle column (tenure/months) found in dataset. "
                "Lifecycle analysis requires a numeric column whose name contains "
                "'tenure', 'month', 'months', 'duration', or 'time' "
                "(columns named 'age', 'id', 'salary', etc. are excluded)."
            )
        }

    # --- Step 2: bucket into lifecycle stages 
    early = df[df[lifecycle_col] <= 3]
    mid   = df[(df[lifecycle_col] > 3) & (df[lifecycle_col] <= 12)]
    late  = df[df[lifecycle_col] > 12]

    early_stats = summarize_actual_churn(early, target_col)
    mid_stats   = summarize_actual_churn(mid,   target_col)
    late_stats  = summarize_actual_churn(late,  target_col)

    segments = {
        "early_stage": early_stats,
        "mid_stage":   mid_stats,
        "late_stage":  late_stats,
    }

    # highest-risk segment (ignore stages with zero customers)
    highest_segment = max(
        segments.items(),
        key=lambda x: x[1]["churn_rate"] if x[1]["customers"] > 0 else 0
    )[0]

    result = {
        "column_used": lifecycle_col,
        "interpretation": (
            f"Lifecycle is calculated using the duration-based column "
            f"'{lifecycle_col}' (values treated as customer duration). "
            f"Early stage: ≤3 , Mid stage: 4–12 , Late stage: >12."
        ),
        "segments": segments,
        "highest_risk_segment": highest_segment,
    }

    # --- Step 3: add predicted churn for early stage if predictions available ---
    if predictions_df is not None and lifecycle_col in predictions_df.columns:
        result["early_stage_predicted"] = summarize_predicted_churn(
            predictions_df[predictions_df[lifecycle_col] <= 3]
        )

    return result