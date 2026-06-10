import pandas as pd
import joblib
from pathlib import Path
from retention_engine import (
    enrich_customer_data,
    build_experiment_summary,
    build_strategy_summary,
)


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"


def format_compact_rs(value):
    absolute_value = abs(value)

    if absolute_value >= 2_000_000:
        formatted = f"{value / 1_000_000:.1f}M"
    elif absolute_value >= 1_000_000:
        formatted = f"{value / 1_000_000:.2f}M"
    elif absolute_value >= 100_000:
        formatted = f"{value / 1_000:.0f}K"
    elif absolute_value >= 1_000:
        formatted = f"{value / 1_000:.1f}K"
    else:
        formatted = f"{value:,.0f}"

    return f"Rs {formatted}"


def load_data():
    return pd.read_csv(
        DATA_DIR / "final_customer_data.csv"
    )

def load_model():

    return joblib.load(
        MODELS_DIR / "churn_rf.pkl"
    )

def load_features():

    return joblib.load(
        MODELS_DIR / "feature_columns.pkl"
    )

def load_reference():

    return joblib.load(
        MODELS_DIR / "reference_customer.pkl"
    )
    
def load_segment_model():
    return joblib.load(
        MODELS_DIR / "kmeans_model.pkl"
    )

def load_segment_columns():
    return joblib.load(
        MODELS_DIR / "segment_columns.pkl"
    )

def load_segment_scaler():
    return joblib.load(
        MODELS_DIR / "kmeans_scaler.pkl"
    )


def load_retention_data():
    return enrich_customer_data(
        load_data(),
        load_model(),
        load_features(),
    )


def load_experiment_summary():
    return build_experiment_summary(
        load_retention_data()
    )


def load_strategy_summary():
    return build_strategy_summary(
        load_experiment_summary()
    )


def load_model_metrics():
    return {
        "Accuracy": 0.88,
        "Precision": 0.76,
        "Recall": 0.85,
        "ROC-AUC": 0.9135289737637422,
        "Test Rows": 10000,
    }


def load_feature_importance(top_n=15):
    model = load_model()
    features = load_features()

    importance = (
        pd.Series(model.feature_importances_, index=features)
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    importance.columns = ["Feature", "Importance"]

    return importance


def explain_customer_risk(customer, population):
    rules = [
        (
            "Inactivity_Risk",
            "high",
            "High inactivity risk",
            "Customer is inactive relative to the customer base.",
        ),
        (
            "Customer_Friction_Index",
            "high",
            "High customer friction",
            "Support calls and cart friction are elevated.",
        ),
        (
            "Loyalty_Score",
            "low",
            "Low loyalty score",
            "Engagement and tenure signals are weaker than peers.",
        ),
        (
            "Cart_Abandonment_Rate",
            "high",
            "High cart abandonment",
            "Customer is abandoning baskets more often than peers.",
        ),
        (
            "Days_Since_Last_Purchase",
            "high",
            "Long time since last purchase",
            "Purchase recency is worse than the portfolio norm.",
        ),
        (
            "Customer_Service_Calls",
            "high",
            "High customer service calls",
            "Service burden is a churn warning signal.",
        ),
        (
            "Customer_Activity_Score",
            "low",
            "Low activity score",
            "Current product activity is below the portfolio norm.",
        ),
    ]

    drivers = []
    for feature, direction, driver, interpretation in rules:
        if feature not in customer or feature not in population.columns:
            continue

        value = float(customer[feature])
        percentile = (population[feature] <= value).mean()
        if direction == "low":
            risk_score = 1 - percentile
            comparison = "below"
        else:
            risk_score = percentile
            comparison = "above"

        if risk_score >= 0.60:
            drivers.append(
                {
                    "Risk Driver": driver,
                    "Feature": feature,
                    "Customer Value": value,
                    "Risk Percentile": risk_score,
                    "Interpretation": interpretation,
                    "Comparison": comparison,
                }
            )

    columns = [
        "Risk Driver",
        "Feature",
        "Customer Value",
        "Risk Percentile",
        "Interpretation",
        "Comparison",
    ]
    if not drivers:
        return pd.DataFrame(columns=columns)

    return (
        pd.DataFrame(drivers, columns=columns)
        .sort_values("Risk Percentile", ascending=False)
        .head(5)
    )
