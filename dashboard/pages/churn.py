import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from utils import (
    explain_customer_risk,
    format_compact_rs,
    load_data,
    load_feature_importance,
    load_model,
    load_features,
    load_reference,
    load_segment_model,
    load_segment_columns,
    load_segment_scaler
)
from retention_engine import STRATEGY_CATALOG, SEGMENT_STRATEGY

population_df = load_data()
rf = load_model()
feature_cols = load_features()
reference = load_reference()
segment_model = load_segment_model()
segment_cols = load_segment_columns()
segment_scaler = load_segment_scaler()
feature_importance = load_feature_importance(top_n=10)


def _safe_z(df, col):
    """Return z-scored center values for a column; zeros if missing/constant."""
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)
    s = df[col].astype(float)
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=df.index)
    return (s - s.mean()) / std


def build_cluster_name_map(model, scaler, columns):
    """Map arbitrary KMeans labels to stable business segment names."""
    centers_scaled = model.cluster_centers_
    try:
        centers_raw = scaler.inverse_transform(centers_scaled)
    except Exception:
        centers_raw = centers_scaled

    centers_df = pd.DataFrame(centers_raw, columns=columns)
    idx = centers_df.index

    frustrated_score = (
        _safe_z(centers_df, "Customer_Friction_Index")
        + _safe_z(centers_df, "Customer_Service_Calls")
        - _safe_z(centers_df, "Customer_Activity_Score")
        - _safe_z(centers_df, "Loyalty_Score")
    )

    champion_score = (
        _safe_z(centers_df, "Customer_Activity_Score")
        + _safe_z(centers_df, "Loyalty_Score")
        + _safe_z(centers_df, "Lifetime_Value")
        + _safe_z(centers_df, "Total_Purchases")
        - _safe_z(centers_df, "Customer_Friction_Index")
        - _safe_z(centers_df, "Inactivity_Risk")
        - _safe_z(centers_df, "Days_Since_Last_Purchase")
    )

    risk_score = (
        _safe_z(centers_df, "Inactivity_Risk")
        + _safe_z(centers_df, "Days_Since_Last_Purchase")
        - _safe_z(centers_df, "Customer_Activity_Score")
        - _safe_z(centers_df, "Loyalty_Score")
    )

    name_map = {i: "Regular Customers" for i in idx}

    champion_idx = int(champion_score.idxmax())
    name_map[champion_idx] = "Champions"

    remaining = [i for i in idx if i != champion_idx]
    if remaining:
        risk_idx = int(risk_score.loc[remaining].idxmax())
        name_map[risk_idx] = "At Risk"
        remaining = [i for i in remaining if i != risk_idx]

    if remaining:
        frustrated_idx = int(frustrated_score.loc[remaining].idxmax())
        name_map[frustrated_idx] = "Frustrated Customers"

    return name_map


def build_cluster_debug_table(model, scaler, columns):
    """Build a transparent table of cluster centers, scores, and mapped labels."""
    centers_scaled = model.cluster_centers_
    try:
        centers_raw = scaler.inverse_transform(centers_scaled)
    except Exception:
        centers_raw = centers_scaled

    centers_df = pd.DataFrame(centers_raw, columns=columns)

    debug_df = pd.DataFrame(index=centers_df.index)
    debug_df["mapped_segment"] = debug_df.index.map(cluster_names)
    debug_df["champion_score"] = (
        _safe_z(centers_df, "Customer_Activity_Score")
        + _safe_z(centers_df, "Loyalty_Score")
        + _safe_z(centers_df, "Lifetime_Value")
        + _safe_z(centers_df, "Total_Purchases")
        - _safe_z(centers_df, "Customer_Friction_Index")
        - _safe_z(centers_df, "Inactivity_Risk")
        - _safe_z(centers_df, "Days_Since_Last_Purchase")
    )
    debug_df["risk_score"] = (
        _safe_z(centers_df, "Inactivity_Risk")
        + _safe_z(centers_df, "Days_Since_Last_Purchase")
        - _safe_z(centers_df, "Customer_Activity_Score")
        - _safe_z(centers_df, "Loyalty_Score")
    )
    debug_df["frustrated_score"] = (
        _safe_z(centers_df, "Customer_Friction_Index")
        + _safe_z(centers_df, "Customer_Service_Calls")
        - _safe_z(centers_df, "Customer_Activity_Score")
        - _safe_z(centers_df, "Loyalty_Score")
    )

    # Include a few raw center values so mapping behavior is easy to inspect.
    for col in [
        "Customer_Activity_Score",
        "Customer_Friction_Index",
        "Customer_Service_Calls",
        "Inactivity_Risk",
        "Days_Since_Last_Purchase",
        "Loyalty_Score",
        "Lifetime_Value",
        "Total_Purchases",
        "Cart_Abandonment_Rate",
    ]:
        if col in centers_df.columns:
            debug_df[col] = centers_df[col]

    return debug_df


cluster_names = build_cluster_name_map(
    segment_model,
    segment_scaler,
    segment_cols
)
cluster_debug = build_cluster_debug_table(
    segment_model,
    segment_scaler,
    segment_cols
)

st.title("Churn Prediction Dashboard")

age = st.slider("Age", 18, 80, 38)

membership_years = st.slider("Membership Years", 0.0, 10.0, 2.5, 0.1)

login_frequency = st.slider("Login Frequency", 0.0, 1.0, 0.24, 0.01)

activity = st.slider("Customer Activity Score", 0.0, 1.0, 0.28, 0.01)

cart = st.slider("Cart Abandonment Rate", 0.0, 145.0, 58.1, 0.1)

lifetime = st.slider("Lifetime Value", 0.0, 9000.0, 1243.0, 10.0)

calls = st.slider("Customer Service Calls", 0, 21, 5)

days = st.slider("Days Since Last Purchase", 0, 287, 21)

purchases = st.slider("Total Purchases", 0, 130, 12)

discount = st.slider("Discount Usage Rate", 0.0, 120.0, 40.2, 0.1)

avg_order = st.slider("Average Order Value", 0.0, 1000.0, 113.0, 1.0)

friction = calls + cart / 10
inactivity = days / (login_frequency + 1)
revenue = lifetime / (purchases + 1)
discount_dependency = discount * purchases
loyalty = (
    0.4 * membership_years
    + 0.3 * login_frequency
    + 0.3 * purchases
)

customer = reference.copy()

customer["Age"] = age
customer["Membership_Years"] = membership_years
customer["Login_Frequency"] = login_frequency
customer["Cart_Abandonment_Rate"] = cart
customer["Customer_Friction_Index"] = friction
customer["Customer_Activity_Score"] = activity
customer["Lifetime_Value"] = lifetime
customer["Customer_Service_Calls"] = calls
customer["Inactivity_Risk"] = inactivity
customer["Days_Since_Last_Purchase"] = days
customer["Total_Purchases"] = purchases
customer["Discount_Usage_Rate"] = discount
customer["Discount_Dependency"] = discount_dependency
customer["Revenue_Per_Purchase"] = revenue
customer["Average_Order_Value"] = avg_order
customer["Loyalty_Score"] = loyalty

customer_df = pd.DataFrame([customer])

customer_df = customer_df.reindex(
    columns=feature_cols,
    fill_value=0
)
segment_input = customer_df[
    segment_cols
]

segment_scaled = segment_scaler.transform(
    segment_input
)

cluster = segment_model.predict(
    segment_scaled
)[0]
prob = rf.predict_proba(customer_df)[0][1]

st.metric(
    "Churn Probability",
    f"{prob*100:.1f}%"
)
segment_name = cluster_names.get(cluster, "Regular Customers")
if prob > 0.7:
    st.error("HIGH CHURN RISK")
elif prob > 0.40:
    st.warning("MEDIUM CHURN RISK")
else:
    st.success("LOW CHURN RISK")
st.subheader(
    "Customer Segment"
)

st.info(
    segment_name
)

st.subheader("Top Risk Drivers")
risk_drivers = explain_customer_risk(
    customer,
    population_df,
)

if risk_drivers.empty:
    st.success(
        "No major risk driver stands out against the customer base. "
        "The model score is likely coming from a broader combination of signals."
    )
else:
    for _, driver in risk_drivers.iterrows():
        st.markdown(
            f"- **{driver['Risk Driver']}** "
            f"({driver['Risk Percentile']:.0%} risk percentile): "
            f"{driver['Interpretation']}"
        )

    driver_fig = px.bar(
        risk_drivers.sort_values("Risk Percentile"),
        x="Risk Percentile",
        y="Risk Driver",
        orientation="h",
        title="Customer-Specific Risk Driver Strength",
        labels={
            "Risk Percentile": "Risk Percentile",
            "Risk Driver": "Risk Driver",
        },
    )
    driver_fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(driver_fig, use_container_width=True)

with st.expander("Global model feature importance"):
    importance_fig = px.bar(
        feature_importance.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top RandomForest Features",
    )
    st.plotly_chart(importance_fig, use_container_width=True)

recommended_intervention = (
    "Win-Back Campaign"
    if segment_name == "At Risk" and prob >= 0.65
    else SEGMENT_STRATEGY.get(
        segment_name,
        "Personalized Offers Campaign",
    )
)
strategy_details = STRATEGY_CATALOG[recommended_intervention]
expected_uplift = strategy_details["expected_uplift"]
campaign_cost = strategy_details["cost_per_customer"]
revenue_saved = lifetime * expected_uplift
net_gain = revenue_saved - campaign_cost
roi = (
    net_gain / campaign_cost * 100
    if campaign_cost > 0
    else 0
)

with st.expander("Debug: segment mapping details"):
    st.write("Current cluster id:", int(cluster))
    st.write("Mapped segment:", segment_name)
    st.write("Active cluster -> segment map:")
    st.json({int(k): v for k, v in cluster_names.items()})
    st.write("Cluster center diagnostics (higher scores indicate stronger profile match):")
    st.dataframe(cluster_debug.round(3))

st.subheader("Recommended Actions")

if segment_name == "Champions":

    if prob > 0.6:
        recs = [
            "VIP retention offer",
            "Dedicated account manager",
            "Priority support"
        ]
    else:
        recs = [
            "Loyalty rewards",
            "Early access promotions"
        ]

elif segment_name == "At Risk":

    recs = [
        "Win-back campaign",
        "Personalized discount",
        "Re-engagement email sequence"
    ]

elif segment_name == "Frustrated Customers":

    recs = [
        "Customer success call",
        "Support escalation",
        "Issue resolution campaign"
    ]

else:   # Regular Customers

    recs = [
        "Personalized product offers",
        "Next-best-action campaign"
    ]

for rec in recs:
    st.markdown(f"- {rec}")

st.subheader("Recommended Retention Strategy")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Intervention",
    recommended_intervention
)

col2.metric(
    "Expected Uplift",
    f"{expected_uplift:.1%}"
)

col3.metric(
    "Expected ROI",
    f"{roi:.1f}%"
)

st.caption(
    strategy_details["description"]
)

st.subheader("Business Impact")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Revenue Saved",
    format_compact_rs(revenue_saved)
)

col2.metric(
    "Campaign Cost",
    format_compact_rs(campaign_cost)
)

col3.metric(
    "Net Gain",
    format_compact_rs(net_gain)
)


st.subheader("Current Customer Profile")

st.dataframe(
    customer_df[
        [
            "Age",
            "Membership_Years",
            "Login_Frequency",
            "Cart_Abandonment_Rate",
            "Customer_Friction_Index",
            "Customer_Activity_Score",
            "Lifetime_Value",
            "Customer_Service_Calls",
            "Inactivity_Risk",
            "Days_Since_Last_Purchase",
            "Total_Purchases",
            "Discount_Usage_Rate",
            "Discount_Dependency",
            "Revenue_Per_Purchase",
            "Average_Order_Value",
            "Loyalty_Score"
        ]
    ]
)

fig = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=prob*100,
        title={"text":"Churn Risk"},
        gauge={
            "axis":{"range":[0,100]},
            "steps":[
                {"range":[0,40]},
                {"range":[40,75]},
                {"range":[75,100]}
            ]
        }
    )
)

st.plotly_chart(fig, use_container_width=True)
