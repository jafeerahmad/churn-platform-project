"""Streamlit dashboard. Run with `streamlit run app/dashboard.py`."""
import sys
from datetime import timedelta
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import text

from src import bootstrap, simulator
from src.config import utcnow
from src.db import engine
from src.features import FEATURES, risk_level
from src.predict import load_metrics, score

st.set_page_config(page_title="Customer Analytics Platform", layout="wide")
st.title("Real-Time Customer Analytics and Churn Prediction")
st.caption(
    "Customer activity events are simulated. The churn model uses a synthetic "
    "Telco-style sample unless you supply your own compatible dataset."
)


@st.cache_resource(show_spinner="Preparing the data, model, and event history...")
def boot() -> bool:
    bootstrap.ensure_ready()
    return True


boot()


@st.cache_resource
def customer_ids():
    return simulator.customer_ids()


model_metrics = load_metrics()
with st.sidebar:
    st.header("Settings")
    live = st.toggle("Auto-refresh every 5 seconds", value=True)
    generate_live = st.toggle(
        "Simulate live traffic", value=True,
        help="Adds a small batch of simulated events on each refresh.",
    )
    if model_metrics:
        st.subheader("Model summary")
        st.write(f"Selected model: **{model_metrics['best_model']}**")
        st.json(model_metrics["results"][model_metrics["best_model"]])


@st.cache_data(ttl=60)
def scored_customers() -> pd.DataFrame:
    customers = pd.read_sql("SELECT * FROM customers", engine)
    customers["churn_probability"] = score(customers[FEATURES])
    customers["risk"] = customers["churn_probability"].map(risk_level)
    return customers


def read_events(days: int) -> pd.DataFrame:
    since = utcnow() - timedelta(days=days)
    query = text(
        "SELECT customer_id, event_type, amount, event_time "
        "FROM events WHERE event_time >= :since"
    )
    return pd.read_sql(
        query, engine, params={"since": since}, parse_dates=["event_time"]
    )


def live_panel():
    if generate_live:
        simulator.tick(customer_ids())
    events = read_events(7)
    if events.empty:
        st.info("No events are available for the selected period.")
        return

    now = utcnow()
    last_24h = events[events.event_time >= now - timedelta(days=1)]
    today = events[events.event_time.dt.date == now.date()]
    orders = events[events.event_type == "order"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Events (24 hours)", f"{len(last_24h):,}")
    col2.metric("Active users today", f"{today.customer_id.nunique():,}")
    col3.metric("Simulated order value (7 days)", f"{orders.amount.sum():,.0f} units")
    col4.metric("Simulated cancellations (7 days)", int((events.event_type == "cancellation").sum()))

    left, right = st.columns(2)
    hourly = (
        last_24h.set_index("event_time").groupby("event_type")
        .resample("1h").size().rename("events").reset_index()
    )
    left.plotly_chart(
        px.bar(hourly, x="event_time", y="events", color="event_type",
               title="Simulated events per hour (last 24 hours)"),
        use_container_width=True,
    )
    daily = (
        orders.set_index("event_time").amount.resample("1D").sum().reset_index()
    )
    right.plotly_chart(
        px.line(daily, x="event_time", y="amount", markers=True,
                title="Simulated daily order value (last 7 days)"),
        use_container_width=True,
    )

    st.subheader("Latest simulated events")
    st.dataframe(
        events.sort_values("event_time", ascending=False).head(10),
        hide_index=True, use_container_width=True,
    )


# A fragment refreshes the activity panel without rerunning the full page.
st.fragment(run_every=5 if live else None)(live_panel)()

st.divider()
st.subheader("High-risk customers")
customers = scored_customers()
recent_events = read_events(7)
tickets = (
    recent_events[recent_events.event_type == "support_ticket"]
    .groupby("customer_id").size().rename("support_tickets_7d")
)
customers = customers.merge(
    tickets, left_on="customerID", right_index=True, how="left"
)
customers["support_tickets_7d"] = customers["support_tickets_7d"].fillna(0).astype(int)

left, right = st.columns([1, 2])
left.plotly_chart(
    px.pie(
        customers, names="risk", title="Customers by predicted churn risk",
        color="risk", color_discrete_map={
            "High": "#e74c3c", "Medium": "#f1c40f", "Low": "#2ecc71",
        },
    ),
    use_container_width=True,
)
top_customers = customers.sort_values("churn_probability", ascending=False).head(15)
right.dataframe(
    top_customers[["customerID", "Contract", "tenure", "MonthlyCharges",
                   "support_tickets_7d", "churn_probability"]],
    hide_index=True, use_container_width=True,
)

st.divider()
st.subheader("Try a churn prediction")
with st.form("predict"):
    col1, col2, col3 = st.columns(3)
    tenure = col1.slider("Tenure (months)", 0, 72, 6)
    monthly = col1.number_input("Monthly charges", 18.0, 120.0, 85.0)
    contract = col2.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = col2.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    payment = col3.selectbox(
        "Payment method",
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
    )
    senior = col3.checkbox("Senior citizen")
    submitted = st.form_submit_button("Predict churn risk")

if submitted:
    row = pd.DataFrame([{
        "SeniorCitizen": int(senior), "tenure": tenure,
        "MonthlyCharges": monthly, "TotalCharges": monthly * tenure,
        "Partner": "No", "Dependents": "No", "InternetService": internet,
        "Contract": contract, "PaperlessBilling": "Yes", "PaymentMethod": payment,
    }])
    probability = float(score(row)[0])
    st.metric("Predicted churn probability", f"{probability:.1%}")
    st.write(f"Risk level: **{risk_level(probability)}**")
    st.progress(min(probability, 1.0))

if model_metrics:
    with st.expander("Top model features"):
        feature_importance = pd.DataFrame(
            model_metrics["top_features"], columns=["feature", "importance"]
        )
        st.plotly_chart(
            px.bar(
                feature_importance.iloc[::-1], x="importance", y="feature",
                orientation="h", title="Top features by model importance",
            ),
            use_container_width=True,
        )
