import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import date
from train_model import build_model
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "expense_category_model.joblib"
DATA_PATH = ROOT / "data" / "expenses.csv"

st.set_page_config(page_title="AI Expense Predictor", page_icon="💸", layout="wide")

CATEGORIES = ["Food", "Transport", "Shopping", "Bills", "Entertainment", "Health", "Education", "Travel", "Other"]

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    training_data = pd.read_csv(ROOT / "data" / "training_expenses.csv")
    return build_model(training_data)

def load_expenses():
    if DATA_PATH.exists():
        try:
            df = pd.read_csv(DATA_PATH)
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
                df = df.dropna(subset=["date", "amount", "description"])
                return df
        except Exception:
            pass
    return pd.DataFrame(columns=["date", "description", "amount", "category"])

def save_expenses(df):
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(DATA_PATH, index=False)

def predict_category(description, model):
    if model is None:
        return "Other", 0.0
    probabilities = model.predict_proba([description])[0]
    index = int(np.argmax(probabilities))
    return model.classes_[index], float(probabilities[index])

def monthly_forecast(df):
    if df.empty:
        return None, None
    monthly = df.copy()
    monthly["month"] = monthly["date"].dt.to_period("M").dt.to_timestamp()
    totals = monthly.groupby("month")["amount"].sum().sort_index()
    if len(totals) == 0:
        return None, None
    # Fill missing months so the trend is based on a continuous timeline.
    full_index = pd.date_range(totals.index.min(), totals.index.max(), freq="MS")
    totals = totals.reindex(full_index, fill_value=0)
    if len(totals) == 1:
        prediction = float(totals.iloc[-1])
    else:
        x = np.arange(len(totals))
        slope, intercept = np.polyfit(x, totals.values, 1)
        prediction = max(0.0, float(slope * len(totals) + intercept))
    return totals, prediction


st.title("💸 AI Expense Category Predictor")
st.caption("Track spending, automatically classify transactions, and estimate next month's expenses.")

model = load_model()

df = load_expenses()
uploaded_expenses = None
skipped_upload_rows = 0

with st.sidebar:
    st.header("➕ Add an expense")
    with st.form("add_expense", clear_on_submit=True):
        expense_date = st.date_input("Date", value=date.today())
        description = st.text_input("What did you spend on?", placeholder="e.g. lunch at cafe")
        amount = st.number_input("Amount (₹)", min_value=1.0, value=150.0, step=10.0)
        submitted = st.form_submit_button("Predict category & save", use_container_width=True)
        if submitted:
            if not description.strip():
                st.warning("Please enter an expense description.")
            else:
                category, confidence = predict_category(description.strip(), model)
                new_row = pd.DataFrame([{
                    "date": pd.Timestamp(expense_date),
                    "description": description.strip(),
                    "amount": float(amount),
                    "category": category
                }])
                df = pd.concat([df, new_row], ignore_index=True)
                save_expenses(df)
                st.success(f"Saved as {category} ({confidence:.0%} model confidence).")
                st.rerun()

    st.divider()
    st.subheader("📂 Import transactions")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"], help="Required columns: date, description, amount. Category is optional.")
    if uploaded is not None:
        try:
            uploaded.seek(0)
            incoming = pd.read_csv(uploaded)
            required = {"date", "description", "amount"}
            if not required.issubset(incoming.columns):
                st.error("CSV must contain: date, description, amount.")
            else:
                row_count = len(incoming)
                incoming["date"] = pd.to_datetime(incoming["date"], errors="coerce")
                incoming["amount"] = pd.to_numeric(incoming["amount"], errors="coerce")
                incoming["description"] = incoming["description"].astype("string").str.strip()
                incoming = incoming.dropna(subset=["date", "description", "amount"])
                incoming = incoming[incoming["description"] != ""].copy()
                skipped_upload_rows = row_count - len(incoming)
                if "category" not in incoming.columns:
                    incoming["category"] = incoming["description"].astype(str).apply(lambda x: predict_category(x, model)[0])
                else:
                    missing_cat = incoming["category"].isna() | (incoming["category"].astype(str).str.strip() == "")
                    incoming.loc[missing_cat, "category"] = incoming.loc[missing_cat, "description"].astype(str).apply(lambda x: predict_category(x, model)[0])
                uploaded_expenses = incoming[["date", "description", "amount", "category"]]
                if uploaded_expenses.empty:
                    st.warning("No valid transactions were found in this CSV.")
                elif st.button("Import transactions", use_container_width=True):
                    df = pd.concat([df, uploaded_expenses], ignore_index=True)
                    save_expenses(df)
                    st.success(f"Imported {len(uploaded_expenses)} transactions.")
                    st.rerun()
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    if st.button("Load demo transactions", use_container_width=True):
        demo_path = ROOT / "data" / "sample_expenses.csv"
        demo = pd.read_csv(demo_path)
        demo["date"] = pd.to_datetime(demo["date"])
        df = pd.concat([df, demo], ignore_index=True)
        save_expenses(df)
        st.success("Demo transactions added.")
        st.rerun()

    if st.button("Clear all transactions", type="secondary", use_container_width=True):
        save_expenses(pd.DataFrame(columns=["date", "description", "amount", "category"]))
        st.rerun()

if uploaded_expenses is not None and not uploaded_expenses.empty:
    st.subheader("📊 Analysis of uploaded data")
    total_uploaded_spend = float(uploaded_expenses["amount"].sum())
    average_uploaded_spend = float(uploaded_expenses["amount"].mean())
    median_uploaded_spend = float(uploaded_expenses["amount"].median())
    category_totals = uploaded_expenses.groupby("category")["amount"].sum().sort_values(ascending=False)
    top_category = category_totals.index[0]
    top_category_spend = float(category_totals.iloc[0])
    largest_transaction = uploaded_expenses.loc[uploaded_expenses["amount"].idxmax()]
    date_range = (
        uploaded_expenses["date"].min().strftime("%d %b %Y"),
        uploaded_expenses["date"].max().strftime("%d %b %Y"),
    )

    analysis_metrics = st.columns(4)
    analysis_metrics[0].metric("Uploaded transactions", len(uploaded_expenses))
    analysis_metrics[1].metric("Total spending", f"₹{total_uploaded_spend:,.2f}")
    analysis_metrics[2].metric("Average transaction", f"₹{average_uploaded_spend:,.2f}")
    analysis_metrics[3].metric("Median transaction", f"₹{median_uploaded_spend:,.2f}")
    st.caption(f"Date range: {date_range[0]} – {date_range[1]}")

    if total_uploaded_spend > 0:
        category_share = top_category_spend / total_uploaded_spend * 100
        st.info(
            f"Highest spending category: **{top_category}** "
            f"(₹{top_category_spend:,.2f}, {category_share:.1f}% of total). "
            f"Largest transaction: **{largest_transaction['description']}** "
            f"(₹{largest_transaction['amount']:,.2f} on "
            f"{largest_transaction['date'].strftime('%d %b %Y')})."
        )
    else:
        st.info(
            f"Highest spending category: **{top_category}** "
            f"(₹{top_category_spend:,.2f}). Largest transaction: "
            f"**{largest_transaction['description']}** "
            f"(₹{largest_transaction['amount']:,.2f} on "
            f"{largest_transaction['date'].strftime('%d %b %Y')})."
        )

    analysis_left, analysis_right = st.columns(2)
    with analysis_left:
        uploaded_category_totals = category_totals.rename_axis("category").reset_index(name="amount")
        category_fig = px.bar(
            uploaded_category_totals,
            x="category",
            y="amount",
            labels={"category": "Category", "amount": "Spending (₹)"},
            title="Spending by category",
        )
        st.plotly_chart(category_fig, use_container_width=True)
    with analysis_right:
        uploaded_monthly = uploaded_expenses.assign(
            month=uploaded_expenses["date"].dt.to_period("M").dt.to_timestamp()
        ).groupby("month", as_index=False)["amount"].sum()
        monthly_fig = px.line(
            uploaded_monthly,
            x="month",
            y="amount",
            markers=True,
            labels={"month": "Month", "amount": "Spending (₹)"},
            title="Monthly spending",
        )
        st.plotly_chart(monthly_fig, use_container_width=True)

    if skipped_upload_rows:
        st.warning(f"{skipped_upload_rows} invalid or incomplete row(s) were excluded from the analysis.")

if df.empty:
    st.info("No transactions yet. Add an expense from the left panel or load the demo transactions to explore the dashboard.")
    st.stop()

df["date"] = pd.to_datetime(df["date"])
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
df = df.dropna(subset=["date", "amount"])

today = pd.Timestamp.today()
this_month = df[(df["date"].dt.year == today.year) & (df["date"].dt.month == today.month)]
total_spend = float(df["amount"].sum())
month_spend = float(this_month["amount"].sum())
avg_transaction = float(df["amount"].mean())
category_count = int(df["category"].nunique())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total recorded spending", f"₹{total_spend:,.0f}")
c2.metric("This month", f"₹{month_spend:,.0f}")
c3.metric("Average transaction", f"₹{avg_transaction:,.0f}")
c4.metric("Categories used", category_count)

left, right = st.columns(2)
with left:
    st.subheader("Spending by category")
    category_totals = df.groupby("category", as_index=False)["amount"].sum().sort_values("amount", ascending=False)
    fig = px.pie(category_totals, names="category", values="amount", hole=0.48)
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.subheader("Monthly spending")
    monthly, forecast = monthly_forecast(df)
    if monthly is not None:
        chart_df = monthly.rename_axis("month").reset_index(name="spending")
        fig2 = px.line(chart_df, x="month", y="spending", markers=True, labels={"month": "Month", "spending": "Spending (₹)"})
        next_month = monthly.index[-1] + pd.offsets.MonthBegin(1)
        fig2.add_scatter(x=[monthly.index[-1], next_month], y=[monthly.iloc[-1], forecast], mode="lines+markers", name="Forecast", line=dict(dash="dash"))
        st.plotly_chart(fig2, use_container_width=True)

if monthly is not None:
    st.subheader("🔮 Next-month spending estimate")
    st.metric("Predicted next-month spending", f"₹{forecast:,.0f}")
    st.caption("This is a simple trend-based estimate from your recorded monthly totals, not a guarantee. More months of data generally make the estimate more meaningful.")

st.subheader("🧠 Try the AI category predictor")
test_description = st.text_input("Enter an expense description", placeholder="e.g. Uber ride to college", key="test_predict")
if test_description.strip():
    predicted, confidence = predict_category(test_description.strip(), model)
    st.success(f"Predicted category: **{predicted}**  ·  Model confidence: **{confidence:.0%}**")

st.subheader("Transaction history")
display_df = df.sort_values("date", ascending=False).copy()
display_df["date"] = display_df["date"].dt.strftime("%d %b %Y")
st.dataframe(display_df[["date", "description", "amount", "category"]], use_container_width=True, hide_index=True)

with st.expander("Change an expenditure category"):
    category_options = sorted(set(CATEGORIES).union(df["category"].dropna().astype(str)))
    selected_index = st.selectbox(
        "Select an expenditure",
        options=df.index,
        format_func=lambda index: (
            f"{df.at[index, 'date'].strftime('%d %b %Y')} — "
            f"{df.at[index, 'description']} (₹{df.at[index, 'amount']:,.2f})"
        ),
    )
    current_category = df.at[selected_index, "category"]
    if pd.isna(current_category) or not str(current_category).strip():
        current_category = "Other"
    else:
        current_category = str(current_category)
    selected_category = st.selectbox(
        "Category",
        options=category_options,
        index=category_options.index(current_category),
        key=f"edit_category_{selected_index}",
    )
    if st.button("Save category change"):
        if selected_category == current_category:
            st.info("Choose a different category to update this expenditure.")
        else:
            df.at[selected_index, "category"] = selected_category
            save_expenses(df)
            st.success("Expenditure category updated.")
            st.rerun()

csv = df.copy()
csv["date"] = csv["date"].dt.strftime("%Y-%m-%d")
st.download_button("⬇️ Download transactions as CSV", csv.to_csv(index=False).encode("utf-8"), "my_expenses.csv", "text/csv")
