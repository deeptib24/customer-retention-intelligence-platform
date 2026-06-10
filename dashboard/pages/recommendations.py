import streamlit as st
from utils import format_compact_rs, load_retention_data
from retention_engine import STRATEGY_CATALOG


df = load_retention_data()

st.title("Customer Retention Strategy Center")

customer = st.selectbox(
    "Select Customer",
    df.index,
)

row = df.loc[customer]
strategy_details = STRATEGY_CATALOG[row["Recommended_Intervention"]]

st.subheader("A. Recommended Retention Strategy")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Customer Segment",
    row["Segment"],
)

col2.metric(
    "Recommended Intervention",
    row["Recommended_Intervention"],
)

col3.metric(
    "Expected Uplift",
    f"{row['Expected_Uplift']:.1%}",
)

st.info(
    strategy_details["description"]
)

st.divider()

st.subheader("B. Business Impact")

revenue_saved = row["Incremental_Retention_Value"]
campaign_cost = row["Intervention_Cost"]
net_gain = row["Expected_Net_Gain"]
roi = row["ROI_Pct"]

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Revenue Saved",
    format_compact_rs(revenue_saved),
)

col2.metric(
    "Campaign Cost",
    format_compact_rs(campaign_cost),
)

col3.metric(
    "Net Gain",
    format_compact_rs(net_gain),
)

col4.metric(
    "ROI",
    f"{roi:.1f}%",
)

st.divider()

st.subheader("Customer Context")

context_cols = [
    "Lifetime_Value",
    "Churn_Probability",
    "Baseline_Retention_Probability",
    "Treatment_Retention_Probability",
    "Customer_Friction_Index",
    "Cart_Abandonment_Rate",
    "Inactivity_Risk",
    "Days_Since_Last_Purchase",
    "Customer_Service_Calls",
    "Loyalty_Score",
]

st.dataframe(
    df.loc[[customer], context_cols].style.format(
        {
            "Lifetime_Value": "Rs {:,.0f}",
            "Churn_Probability": "{:.1%}",
            "Baseline_Retention_Probability": "{:.1%}",
            "Treatment_Retention_Probability": "{:.1%}",
            "Customer_Friction_Index": "{:.2f}",
            "Cart_Abandonment_Rate": "{:.1f}",
            "Inactivity_Risk": "{:.1f}",
            "Days_Since_Last_Purchase": "{:.0f}",
            "Customer_Service_Calls": "{:.0f}",
            "Loyalty_Score": "{:.2f}",
        }
    ),
    use_container_width=True,
)

st.subheader("Why This Strategy")

if row["Segment"] == "Champions":
    reasons = [
        "High value and high engagement customers should be protected with loyalty benefits.",
        "The goal is retention reinforcement rather than heavy discounting.",
    ]
elif row["Segment"] == "At Risk":
    reasons = [
        "Inactive customers need a concrete reason to return.",
        "The strategy targets churn probability through a recovery offer.",
    ]
elif row["Segment"] == "Frustrated Customers":
    reasons = [
        "High support friction is better solved through service recovery than discounts.",
        "Priority support directly addresses the customer pain signal.",
    ]
else:
    reasons = [
        "Regular customers benefit from relevant offers that increase engagement.",
        "The strategy is lower cost and suitable for broad deployment.",
    ]

for reason in reasons:
    st.markdown(f"- {reason}")
