# Customer Retention Decision Platform

## Architecture

Customer data flows through the existing segmentation and churn models, then into a new decision layer:

Customer Data -> Segmentation -> Churn Prediction -> Retention Strategy Recommendation -> Experimentation Engine -> Uplift Analysis -> ROI Analysis -> Dashboard

The churn model and segmentation model are not rebuilt. The new layer uses saved model artifacts and the existing `final_customer_data.csv` output.

## Data Model Additions

The retention layer enriches customer records in memory with:

- `Segment`: business segment name derived from the existing cluster profile.
- `Churn_Probability`: predicted by the saved RandomForest churn model.
- `Baseline_Retention_Probability`: `1 - Churn_Probability`.
- `Recommended_Intervention`: strategy assigned by segment and risk.
- `Expected_Uplift`: simulated retention lift from the strategy catalog.
- `Treatment_Retention_Probability`: baseline retention plus expected uplift.
- `Intervention_Cost`: marginal cost per targeted customer.
- `Incremental_Retention_Value`: expected revenue saved per customer.
- `Expected_Net_Gain`: expected value after intervention cost.
- `ROI_Pct`: expected return on intervention spend.

## Strategy Assignment

- Champions -> Loyalty Rewards Campaign
- At Risk -> Discount Campaign
- High-risk At Risk -> Win-Back Campaign
- Frustrated Customers -> Priority Support Campaign
- Regular Customers -> Personalized Offers Campaign

## Experimentation Design

Control is simulated as no intervention, using baseline retention probability.

Treatment is simulated as the assigned retention strategy, using baseline retention probability plus expected strategy uplift.

Segment-level results aggregate:

- segment size
- average customer lifetime value
- control retention
- treatment retention
- retention uplift
- customers retained
- revenue saved
- campaign cost
- net gain
- ROI
- deployment decision

## Code Files

Created:

- `dashboard/retention_engine.py`: retention strategy catalog, segment mapping, churn scoring, strategy assignment, experiment summary, ROI summary.
- `RETENTION_PLATFORM_ARCHITECTURE.md`: project architecture and interview explanation.

Modified:

- `dashboard/utils.py`: loads `final_customer_data.csv` and exposes retention summaries.
- `dashboard/pages/churn.py`: adds recommended intervention and customer-level ROI for custom profile predictions.
- `dashboard/pages/recommendations.py`: customer strategy center with recommended intervention and business impact.
- `dashboard/pages/experiments.py`: segment uplift, strategy comparison, ROI, deployment recommendation.
- `dashboard/pages/segments.py`: business segment names, churn risk, strategy assignment.
- `dashboard/app.py`: uses final customer data as dashboard source.

