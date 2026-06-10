import streamlit as st
import plotly.express as px
from utils import load_retention_data


df = load_retention_data()

st.title("Customer Segments")

segment_counts = (
    df["Segment"]
    .value_counts()
    .reset_index()
)
segment_counts.columns = ["Segment", "Customers"]

fig = px.bar(
    segment_counts,
    x="Segment",
    y="Customers",
    title="Customer Count by Business Segment",
)

st.plotly_chart(
    fig,
    use_container_width=True,
)

segment_metrics = (
    df.groupby(["Segment", "Recommended_Intervention"])
    .agg(
        Customers=("Segment", "size"),
        Churn_Risk=("Churn_Probability", "mean"),
        Avg_Lifetime_Value=("Lifetime_Value", "mean"),
        Avg_Friction=("Customer_Friction_Index", "mean"),
        Avg_Inactivity=("Inactivity_Risk", "mean"),
        Expected_Uplift=("Expected_Uplift", "mean"),
    )
    .reset_index()
    .sort_values("Churn_Risk", ascending=False)
)

st.dataframe(
    segment_metrics.style.format(
        {
            "Churn_Risk": "{:.1%}",
            "Avg_Lifetime_Value": "Rs {:,.0f}",
            "Avg_Friction": "{:.2f}",
            "Avg_Inactivity": "{:.1f}",
            "Expected_Uplift": "{:.1%}",
        }
    ),
    use_container_width=True,
)

scatter = px.scatter(
    segment_metrics,
    x="Churn_Risk",
    y="Avg_Lifetime_Value",
    size="Customers",
    color="Recommended_Intervention",
    hover_name="Segment",
    title="Segment Risk vs Value",
    labels={
        "Churn_Risk": "Average Churn Probability",
        "Avg_Lifetime_Value": "Average Lifetime Value",
    },
)

scatter.update_xaxes(tickformat=".0%")

st.plotly_chart(
    scatter,
    use_container_width=True,
)
