import streamlit as st
import plotly.express as px
from utils import (
    format_compact_rs,
    load_experiment_summary,
    load_strategy_summary,
)
from retention_engine import build_deployment_recommendation


experiment_df = load_experiment_summary()
strategy_df = load_strategy_summary()

st.title("Retention Experimentation Analytics")

total_revenue_saved = experiment_df["Revenue_Saved"].sum()
total_campaign_cost = experiment_df["Campaign_Cost"].sum()
total_net_gain = experiment_df["Net_Gain"].sum()
total_roi = (
    total_net_gain / total_campaign_cost * 100
    if total_campaign_cost > 0
    else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Revenue Saved",
    format_compact_rs(total_revenue_saved),
)

col2.metric(
    "Campaign Cost",
    format_compact_rs(total_campaign_cost),
)

col3.metric(
    "Net Gain",
    format_compact_rs(total_net_gain),
)

col4.metric(
    "ROI",
    f"{total_roi:.1f}%",
)

st.divider()

st.subheader("Segment-Level Uplift")

display_cols = [
    "Segment",
    "Recommended_Intervention",
    "Segment_Size",
    "Baseline_Retention",
    "Expected Retention After Campaign",
    "Expected_Uplift",
    "Customers_Retained",
    "Revenue_Saved",
    "Campaign_Cost",
    "Net_Gain",
    "ROI_Pct",
    "Decision",
]

st.dataframe(
    experiment_df[display_cols].style.format(
        {
            "Baseline_Retention": "{:.1%}",
            "Expected Retention After Campaign": "{:.1%}",
            "Expected_Uplift": "{:.1%}",
            "Customers_Retained": "{:.0f}",
            "Revenue_Saved": format_compact_rs,
            "Campaign_Cost": format_compact_rs,
            "Net_Gain": format_compact_rs,
            "ROI_Pct": "{:.1f}%",
        }
    ),
    use_container_width=True,
)

uplift_fig = px.bar(
    experiment_df,
    x="Segment",
    y="Expected_Uplift",
    color="Recommended_Intervention",
    title="Expected Retention Uplift by Segment",
    labels={
        "Expected_Uplift": "Retention Uplift",
        "Recommended_Intervention": "Strategy",
    },
)

uplift_fig.update_yaxes(tickformat=".0%")

st.plotly_chart(
    uplift_fig,
    use_container_width=True,
)

st.subheader("Strategy Comparison")

strategy_fig = px.bar(
    strategy_df,
    x="Recommended_Intervention",
    y="Net_Gain",
    color="ROI_Pct",
    title="Expected Net Gain by Retention Strategy",
    labels={
        "Recommended_Intervention": "Strategy",
        "Net_Gain": "Net Gain",
        "ROI_Pct": "ROI %",
    },
)

st.plotly_chart(
    strategy_fig,
    use_container_width=True,
)

st.dataframe(
    strategy_df.style.format(
        {
            "Average_Uplift": "{:.1%}",
            "Revenue_Saved": format_compact_rs,
            "Campaign_Cost": format_compact_rs,
            "Net_Gain": format_compact_rs,
            "ROI_Pct": "{:.1f}%",
        }
    ),
    use_container_width=True,
)

st.subheader("Deployment Recommendation")

st.info(
    build_deployment_recommendation(experiment_df)
)
