# Data dictionary and provenance

## 1. `telco_churn.csv`

This project includes a **synthetic 7,000-customer dataset** generated locally by
`src/generate_data.py`. It is designed to resemble selected fields from the public
Telco Customer Churn dataset, but it is **not the original Kaggle data** and does
not contain real customer information. The bundled file is a 12-column subset used
by the model in this repository. It must not be described as the complete Kaggle
schema.

| Column | Type | Meaning / allowed values |
|---|---|---|
| `customerID` | string | Unique synthetic customer key. |
| `SeniorCitizen` | integer | 1 = senior-citizen flag; 0 = otherwise. |
| `Partner` | category | `Yes` or `No`. |
| `Dependents` | category | `Yes` or `No`. |
| `tenure` | integer | Length of service in months, 0 to 72. |
| `InternetService` | category | `DSL`, `Fiber optic`, or `No`. |
| `Contract` | category | `Month-to-month`, `One year`, or `Two year`. |
| `PaperlessBilling` | category | `Yes` or `No`. |
| `PaymentMethod` | category | `Electronic check`, `Mailed check`, `Bank transfer`, or `Credit card`. |
| `MonthlyCharges` | decimal | Synthetic monthly charge amount. |
| `TotalCharges` | decimal | Synthetic cumulative charge amount; generated from tenure and monthly charges with small random variation. |
| `Churn` | category | Target label: `Yes` or `No`. |

The synthetic target is generated from a seeded probability formula that uses tenure,
contract, internet service, payment method, senior-citizen flag, paperless billing,
and partner status. The generated labels are therefore useful for demonstrating a
machine-learning workflow, not for estimating real-world churn rates.

The generated file contains 7,000 unique customers: 1,971 churned (`Yes`) and 5,029
not churned (`No`), for a 28.16% churn rate. The 80/20 stratified model split uses
5,600 training rows and 1,400 test rows.

### Using the original/public dataset

To evaluate against an external dataset, obtain the data from the
[Kaggle Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
and place a compatible CSV at `data/telco_churn.csv`. Keep the required fields listed
above. Extra columns are ignored by the model. The project does not bundle or claim
ownership of the Kaggle data.

## 2. `events_sample.csv`

A deterministic 30-day export of 45,000 simulated platform events. It is provided
for inspection and chart generation; the live dashboard writes its own event records
to the configured SQL database.

| Column | Type | Meaning / allowed values |
|---|---|---|
| `event_id` | integer | Sequential row identifier in this sample export. |
| `customer_id` | string | References a `customerID` in `telco_churn.csv`. |
| `event_type` | category | `login`, `order`, `support_ticket`, or `cancellation`. |
| `amount` | decimal | Synthetic order amount; zero for non-order events. No real currency is implied. |
| `event_time` | datetime | Simulated event timestamp. The sample uses a fixed historical time window for reproducibility. |

Event-type probabilities and order amounts are synthetic assumptions configured in
`src/simulator.py`. These records are not real transactions, sales, or support tickets.

## 3. `customer_risk_scores.csv`

A derived export with one row per bundled customer. It includes `customerID`,
`Contract`, `tenure`, `MonthlyCharges`, the demonstration `Churn` label, the selected
model's `churn_probability`, and the `risk_level` bucket. Risk buckets are `Low`
(below 0.30), `Medium` (0.30 to below 0.60), and `High` (0.60 or above). Scores are
model outputs for demonstration and should not be used for real customer decisions.

## 4. Database tables

- `customers`: customer feature fields plus the numeric target `Churn` (1 = Yes, 0 = No).
- `events`: simulated event rows written by the event simulator.

The `data/` CSV files are inspectable source/sample exports. The SQLite database file
is generated locally and is intentionally not included in version control.
