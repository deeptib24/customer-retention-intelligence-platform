import numpy as np
import pandas as pd


STRATEGY_CATALOG = {
    "Loyalty Rewards Campaign": {
        "expected_uplift": 0.025,
        "cost_per_customer": 20,
        "description": "Reward points, early access offers, and VIP retention touches.",
    },
    "Discount Campaign": {
        "expected_uplift": 0.061,
        "cost_per_customer": 45,
        "description": "Targeted discount offer for customers with elevated churn risk.",
    },
    "Priority Support Campaign": {
        "expected_uplift": 0.082,
        "cost_per_customer": 45,
        "description": "Proactive service recovery and support escalation.",
    },
    "Personalized Offers Campaign": {
        "expected_uplift": 0.035,
        "cost_per_customer": 15,
        "description": "Product recommendations and contextual purchase incentives.",
    },
    "Win-Back Campaign": {
        "expected_uplift": 0.095,
        "cost_per_customer": 55,
        "description": "High-intent recovery campaign for severely inactive customers.",
    },
}


SEGMENT_STRATEGY = {
    "Champions": "Loyalty Rewards Campaign",
    "At Risk": "Discount Campaign",
    "Frustrated Customers": "Priority Support Campaign",
    "Regular Customers": "Personalized Offers Campaign",
}


def _safe_z(df, col):
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)

    series = df[col].astype(float)
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=df.index)

    return (series - series.mean()) / std


def build_segment_name_map(df):
    centers = (
        df.groupby("Cluster")
        .agg(
            {
                "Customer_Activity_Score": "mean",
                "Loyalty_Score": "mean",
                "Lifetime_Value": "mean",
                "Total_Purchases": "mean",
                "Customer_Friction_Index": "mean",
                "Customer_Service_Calls": "mean",
                "Inactivity_Risk": "mean",
                "Days_Since_Last_Purchase": "mean",
            }
        )
        .sort_index()
    )

    champion_score = (
        _safe_z(centers, "Customer_Activity_Score")
        + _safe_z(centers, "Loyalty_Score")
        + _safe_z(centers, "Lifetime_Value")
        + _safe_z(centers, "Total_Purchases")
        - _safe_z(centers, "Customer_Friction_Index")
        - _safe_z(centers, "Inactivity_Risk")
        - _safe_z(centers, "Days_Since_Last_Purchase")
    )

    at_risk_score = (
        _safe_z(centers, "Inactivity_Risk")
        + _safe_z(centers, "Days_Since_Last_Purchase")
        - _safe_z(centers, "Customer_Activity_Score")
        - _safe_z(centers, "Loyalty_Score")
    )

    frustrated_score = (
        _safe_z(centers, "Customer_Friction_Index")
        + _safe_z(centers, "Customer_Service_Calls")
        - _safe_z(centers, "Customer_Activity_Score")
        - _safe_z(centers, "Loyalty_Score")
    )

    segment_map = {cluster: "Regular Customers" for cluster in centers.index}

    champion_cluster = champion_score.idxmax()
    segment_map[champion_cluster] = "Champions"

    remaining = [cluster for cluster in centers.index if cluster != champion_cluster]
    if remaining:
        at_risk_cluster = at_risk_score.loc[remaining].idxmax()
        segment_map[at_risk_cluster] = "At Risk"
        remaining = [cluster for cluster in remaining if cluster != at_risk_cluster]

    if remaining:
        frustrated_cluster = frustrated_score.loc[remaining].idxmax()
        segment_map[frustrated_cluster] = "Frustrated Customers"

    return segment_map


def score_churn_probability(df, model, feature_cols):
    model_input = pd.get_dummies(df, drop_first=True)
    model_input = model_input.reindex(columns=feature_cols, fill_value=0)
    model_input = model_input.replace([np.inf, -np.inf], np.nan)
    model_input = model_input.fillna(model_input.median(numeric_only=True))
    model_input = model_input.fillna(0)
    return model.predict_proba(model_input)[:, 1]


def assign_retention_strategy(row):
    if row["Segment"] == "At Risk" and row["Churn_Probability"] >= 0.65:
        return "Win-Back Campaign"

    return SEGMENT_STRATEGY.get(
        row["Segment"],
        "Personalized Offers Campaign",
    )


def enrich_customer_data(df, model=None, feature_cols=None):
    enriched = df.copy()
    segment_map = build_segment_name_map(enriched)
    enriched["Segment"] = enriched["Cluster"].map(segment_map)

    if model is not None and feature_cols is not None:
        enriched["Churn_Probability"] = score_churn_probability(
            enriched,
            model,
            feature_cols,
        )
    elif "Churn_Probability" not in enriched.columns:
        enriched["Churn_Probability"] = enriched["Churned"].astype(float)

    enriched["Baseline_Retention_Probability"] = (
        1 - enriched["Churn_Probability"]
    ).clip(0, 1)

    enriched["Recommended_Intervention"] = enriched.apply(
        assign_retention_strategy,
        axis=1,
    )
    enriched["Expected_Uplift"] = enriched["Recommended_Intervention"].map(
        lambda strategy: STRATEGY_CATALOG[strategy]["expected_uplift"]
    )
    enriched["Intervention_Cost"] = enriched["Recommended_Intervention"].map(
        lambda strategy: STRATEGY_CATALOG[strategy]["cost_per_customer"]
    )

    enriched["Treatment_Retention_Probability"] = (
        enriched["Baseline_Retention_Probability"]
        + enriched["Expected_Uplift"]
    ).clip(0, 1)
    enriched["Incremental_Retention_Value"] = (
        enriched["Expected_Uplift"] * enriched["Lifetime_Value"]
    )
    enriched["Expected_Net_Gain"] = (
        enriched["Incremental_Retention_Value"]
        - enriched["Intervention_Cost"]
    )
    enriched["ROI_Pct"] = np.where(
        enriched["Intervention_Cost"] > 0,
        enriched["Expected_Net_Gain"] / enriched["Intervention_Cost"] * 100,
        np.nan,
    )

    return enriched


def build_experiment_summary(enriched):
    grouped = (
        enriched.groupby(["Segment", "Recommended_Intervention"])
        .agg(
            Segment_Size=("Segment", "size"),
            Average_Lifetime_Value=("Lifetime_Value", "mean"),
            Average_Churn_Probability=("Churn_Probability", "mean"),
            Control_Retention=("Baseline_Retention_Probability", "mean"),
            Treatment_Retention=("Treatment_Retention_Probability", "mean"),
            Expected_Uplift=("Expected_Uplift", "mean"),
            Intervention_Cost_Per_Customer=("Intervention_Cost", "mean"),
        )
        .reset_index()
    )

    grouped["Customers_Retained"] = (
        grouped["Segment_Size"] * grouped["Expected_Uplift"]
    )
    grouped["Revenue_Saved"] = (
        grouped["Customers_Retained"] * grouped["Average_Lifetime_Value"]
    )
    grouped["Campaign_Cost"] = (
        grouped["Segment_Size"]
        * grouped["Intervention_Cost_Per_Customer"]
    )
    grouped["Net_Gain"] = (
        grouped["Revenue_Saved"] - grouped["Campaign_Cost"]
    )
    grouped["ROI_Pct"] = np.where(
        grouped["Campaign_Cost"] > 0,
        grouped["Net_Gain"] / grouped["Campaign_Cost"] * 100,
        np.nan,
    )
    grouped["Decision"] = np.where(
        grouped["Net_Gain"] > 0,
        "Deploy",
        "Pilot / Redesign",
    )

    return grouped.sort_values(
        ["ROI_Pct", "Net_Gain"],
        ascending=False,
    )


def build_strategy_summary(experiment_summary):
    return (
        experiment_summary.groupby("Recommended_Intervention")
        .agg(
            Segment_Size=("Segment_Size", "sum"),
            Average_Uplift=("Expected_Uplift", "mean"),
            Revenue_Saved=("Revenue_Saved", "sum"),
            Campaign_Cost=("Campaign_Cost", "sum"),
            Net_Gain=("Net_Gain", "sum"),
        )
        .reset_index()
        .assign(
            ROI_Pct=lambda data: np.where(
                data["Campaign_Cost"] > 0,
                data["Net_Gain"] / data["Campaign_Cost"] * 100,
                np.nan,
            )
        )
        .sort_values("Net_Gain", ascending=False)
    )


def build_deployment_recommendation(experiment_summary):
    deploy = experiment_summary[experiment_summary["Decision"] == "Deploy"]
    if deploy.empty:
        return "Do not deploy globally yet. Redesign costs or pilot narrower offers for high-value customers."

    best = deploy.sort_values("Net_Gain", ascending=False).iloc[0]
    net_gain = best["Net_Gain"]
    if abs(net_gain) >= 1_000_000:
        compact_net_gain = f"Rs {net_gain / 1_000_000:.1f}M"
    elif abs(net_gain) >= 100_000:
        compact_net_gain = f"Rs {net_gain / 1_000:.0f}K"
    else:
        compact_net_gain = f"Rs {net_gain / 1_000:.1f}K"

    return (
        f"Prioritize {best['Recommended_Intervention']} for "
        f"the {best['Segment']} segment. It has the highest expected net gain "
        f"of {compact_net_gain} in this simulation."
    )
