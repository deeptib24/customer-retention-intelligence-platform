import plotly.express as px
import streamlit as st

from retention_engine import build_deployment_recommendation
from utils import (
    format_compact_rs,
    load_data,
    load_experiment_summary,
    load_feature_importance,
    load_model_metrics,
    load_retention_data,
    load_strategy_summary,
)


st.set_page_config(
    page_title="Customer Retention Intelligence Platform",
    layout="wide",
)


raw_df = load_data()
retention_df = load_retention_data()
experiment_df = load_experiment_summary()
strategy_df = load_strategy_summary()
metrics = load_model_metrics()
feature_importance = load_feature_importance(top_n=12)

customers = len(retention_df)
actual_churn_rate = raw_df["Churned"].mean()
predicted_churn_rate = retention_df["Churn_Probability"].mean()
baseline_retention = retention_df["Baseline_Retention_Probability"].mean()
treatment_retention = retention_df["Treatment_Retention_Probability"].mean()
expected_uplift = treatment_retention - baseline_retention

revenue_saved = experiment_df["Revenue_Saved"].sum()
campaign_cost = experiment_df["Campaign_Cost"].sum()
net_gain = experiment_df["Net_Gain"].sum()
roi_pct = net_gain / campaign_cost * 100 if campaign_cost > 0 else 0
deploy_segments = (experiment_df["Decision"] == "Deploy").sum()
high_risk_customers = (retention_df["Churn_Probability"] >= 0.65).sum()

best_segment = experiment_df.sort_values("Net_Gain", ascending=False).iloc[0]
riskiest_segment = (
    experiment_df.sort_values("Average_Churn_Probability", ascending=False)
    .iloc[0]
)

st.title("Customer Retention Decision Platform")
st.caption(
    "Executive overview of churn risk, customer segments, recommended "
    "interventions, experiment uplift, and expected ROI."
)

st.info(build_deployment_recommendation(experiment_df))

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Customers Analyzed",
    f"{customers:,}",
    f"{high_risk_customers:,} high-risk",
)
col2.metric(
    "Observed Churn",
    f"{actual_churn_rate:.1%}",
)
col3.metric(
    "Predicted Churn Risk",
    f"{predicted_churn_rate:.1%}",
)
col4.metric(
    "Expected Retention Lift",
    f"{expected_uplift:.1%}",
    f"{baseline_retention:.1%} -> {treatment_retention:.1%}",
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Revenue Saved", format_compact_rs(revenue_saved))
col2.metric("Campaign Cost", format_compact_rs(campaign_cost))
col3.metric("Expected Net Gain", format_compact_rs(net_gain))
col4.metric("Portfolio ROI", f"{roi_pct:.1f}%", f"{deploy_segments} deploy segments")

st.divider()

left, right = st.columns([1.15, 0.85])

with left:
    st.subheader("Where the Retention Opportunity Is")

    segment_view = experiment_df.sort_values("Net_Gain", ascending=False)
    segment_fig = px.bar(
        segment_view,
        x="Segment",
        y="Net_Gain",
        color="Recommended_Intervention",
        text="Decision",
        title="Expected Net Gain by Segment and Intervention",
        labels={
            "Net_Gain": "Expected Net Gain",
            "Recommended_Intervention": "Intervention",
        },
    )
    segment_fig.update_layout(barmode="stack")
    segment_fig.update_yaxes(tickprefix="Rs ", tickformat="~s")
    st.plotly_chart(segment_fig, use_container_width=True)

with right:
    st.subheader("Executive Readout")
    st.markdown(
        f"""
- Best value pool: **{best_segment["Segment"]}** via **{best_segment["Recommended_Intervention"]}**.
- Highest churn risk: **{riskiest_segment["Segment"]}** at **{riskiest_segment["Average_Churn_Probability"]:.1%}** average predicted churn.
- Current plan keeps **{experiment_df["Customers_Retained"].sum():,.0f}** incremental customers in expectation.
- This home page is useful as the control room; detailed diagnosis lives in the sidebar pages.
"""
    )

    st.dataframe(
        experiment_df[
            [
                "Segment",
                "Recommended_Intervention",
                "Segment_Size",
                "Expected_Uplift",
                "Net_Gain",
                "ROI_Pct",
                "Decision",
            ]
        ].style.format(
            {
                "Segment_Size": "{:,}",
                "Expected_Uplift": "{:.1%}",
                "Net_Gain": format_compact_rs,
                "ROI_Pct": "{:.1f}%",
            }
        ),
        use_container_width=True,
    )

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn Model Baseline")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Accuracy", f"{metrics['Accuracy']:.2%}")
    metric_cols[1].metric("Precision", f"{metrics['Precision']:.2%}")
    metric_cols[2].metric("Recall", f"{metrics['Recall']:.2%}")
    metric_cols[3].metric("ROC-AUC", f"{metrics['ROC-AUC']:.3f}")
    st.caption(
        f"Evaluation is from the held-out test split in "
        f"`notebooks/03churn_prediction.ipynb` ({metrics['Test Rows']:,} rows). "
        "Precision and recall are reported for the churn class."
    )

with col2:
    st.subheader("Top Churn Risk Signals")
    importance_fig = px.bar(
        feature_importance.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        title="RandomForest Feature Importance",
        labels={"Importance": "Importance", "Feature": "Feature"},
    )
    st.plotly_chart(importance_fig, use_container_width=True)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Campaign Mix")
    strategy_fig = px.bar(
        strategy_df,
        x="Recommended_Intervention",
        y="Segment_Size",
        color="ROI_Pct",
        title="Customers Targeted by Recommended Intervention",
        labels={
            "Recommended_Intervention": "Intervention",
            "Segment_Size": "Customers",
            "ROI_Pct": "ROI %",
        },
    )
    st.plotly_chart(strategy_fig, use_container_width=True)

with col2:
    st.subheader("Risk vs Value")
    risk_value = (
        retention_df.groupby(["Segment", "Recommended_Intervention"])
        .agg(
            Customers=("Segment", "size"),
            Churn_Risk=("Churn_Probability", "mean"),
            Lifetime_Value=("Lifetime_Value", "mean"),
            Net_Gain=("Expected_Net_Gain", "sum"),
        )
        .reset_index()
    )
    scatter_fig = px.scatter(
        risk_value,
        x="Churn_Risk",
        y="Lifetime_Value",
        size="Customers",
        color="Recommended_Intervention",
        hover_name="Segment",
        title="Segment Risk and Customer Value",
        labels={
            "Churn_Risk": "Average Churn Risk",
            "Lifetime_Value": "Average Lifetime Value",
        },
    )
    scatter_fig.update_xaxes(tickformat=".0%")
    scatter_fig.update_yaxes(tickprefix="Rs ", tickformat="~s")
    st.plotly_chart(scatter_fig, use_container_width=True)

st.divider()

st.subheader("Use the Deeper Pages For")
col1, col2, col3, col4 = st.columns(4)

col1.markdown("**Churn**  \nScore a custom customer profile and see segment-level actions.")
col2.markdown("**Segments**  \nCompare business segments by size, risk, value, and intervention.")
col3.markdown("**Recommendations**  \nInspect a customer-level retention strategy and ROI estimate.")
col4.markdown("**Experiments**  \nReview simulated uplift, campaign cost, net gain, and deploy decisions.")
