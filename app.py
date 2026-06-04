import streamlit as st
import pandas as pd
import joblib

st.set_page_config(layout="wide")

st.title("Electricity Peak Demand Prediction System")

# ============================================
# LOAD DATA
# ============================================

daily = pd.read_csv("data/processed/daily_predictions.csv")
hourly_full = pd.read_csv("data/processed/hourly_data.csv")
top_hours = pd.read_csv("data/processed/top_peak_hours.csv")

hourly_full["Datetime"] = pd.to_datetime(hourly_full["Datetime"])
top_hours["Datetime"] = pd.to_datetime(top_hours["Datetime"])
daily["Date"] = pd.to_datetime(daily["Date"]).dt.date

# Load model
peak_hour_model = joblib.load("models/peak_hour_model.pkl")

# ============================================
# DATE SELECTOR (Original Dataset Range)
# ============================================

st.sidebar.header("Select Date")

available_dates = sorted(daily["Date"].unique())

selected_date = st.sidebar.date_input(
    "Choose a date",
    value=available_dates[-1],
    min_value=available_dates[0],
    max_value=available_dates[-1]
)

# ============================================
# PEAK DAY ANALYSIS
# ============================================

st.header("Peak Day Analysis")

selected_day = daily[daily["Date"] == selected_date]

if not selected_day.empty:
    prob = selected_day.iloc[0]["Peak_Day_Prob"]

    st.write("Date:", selected_date)
    st.write(f"Peak Probability: {prob:.6f} ({prob*100:.2f}%)")

    if prob > 0.6:
        st.error("⚠️ High Risk: Possible Peak Day")
    elif prob > 0.3:
        st.warning("Moderate Peak Risk")
    else:
        st.success("Normal Demand Expected")

    # Grid Stress Indicator
    st.subheader("System Stress Level")
    if prob > 0.6:
        st.error("Grid Stress: HIGH")
    elif prob > 0.3:
        st.warning("Grid Stress: MODERATE")
    else:
        st.success("Grid Stress: NORMAL")

else:
    st.warning("No data available for this date.")


# ============================================
# LOAD CURVE
# ============================================

st.header("Load Curve")

day_data = hourly_full[
    hourly_full["Datetime"].dt.date == selected_date
].copy()

if not day_data.empty:
    st.line_chart(
        day_data.set_index("Datetime")["Load"]
    )
else:
    st.warning("No hourly data available for this date.")


# ============================================
# DEMAND-BASED COST RISK
# ============================================

st.header("Demand-Based Cost Risk Index")

if not day_data.empty:

    avg_load = day_data["Load"].mean()

    day_data["Load_Ratio"] = day_data["Load"] / avg_load
    day_data["Cost_Index"] = 1 + 0.6 * (day_data["Load_Ratio"] - 1)

    avg_cost = day_data["Cost_Index"].mean()
    max_cost = day_data["Cost_Index"].max()

    st.write(f"Average Load: {avg_load:.0f}")
    st.write(f"Average Cost Index: {avg_cost:.2f}")
    st.write(f"Peak Cost Index: {max_cost:.2f}")

    if max_cost > 1.5:
        st.error("⚠️ High Cost Risk")
    elif max_cost > 1.25:
        st.warning("Moderate Cost Risk")
    else:
        st.success("Normal Cost Level")

    st.subheader("Hourly Cost Risk Curve")
    st.line_chart(
        day_data.set_index("Datetime")["Cost_Index"]
    )

else:
    st.warning("No data available for cost estimation.")


# ============================================
# TOP PEAK HOURS
# ============================================

st.header("Top Predicted Peak Hours")

top_today = top_hours[
    top_hours["Datetime"].dt.date == selected_date
].sort_values("Peak_Hour_Prob", ascending=False)

if not top_today.empty:

    for _, row in top_today.head(2).iterrows():
        st.write(
            f"Hour: {row['Datetime'].strftime('%H:%M')} | Probability: {row['Peak_Hour_Prob']:.2f}"
        )

    peak_hour = top_today.iloc[0]["Datetime"].strftime("%H:%M")
    st.info(f"Expected Peak Window around: {peak_hour}")

else:
    st.warning("No peak hour predictions for this date.")


# ============================================
# FULL DAY PEAK HOUR PROBABILITY
# ============================================

st.header("Peak Hour Probability Chart (Full Day)")

if not day_data.empty:

    # Create features
    day_data["Hour"] = day_data["Datetime"].dt.hour
    day_data["Month"] = day_data["Datetime"].dt.month
    day_data["IsWeekend"] = day_data["Datetime"].dt.weekday >= 5

    features = [
        "Load",
        "Temperature",
        "Humidity",
        "Hour",
        "Month",
        "IsWeekend"
    ]

    day_data["Peak_Hour_Prob"] = peak_hour_model.predict_proba(
        day_data[features]
    )[:, 1]

    st.line_chart(
        day_data.set_index("Datetime")["Peak_Hour_Prob"]
    )

    # Critical hours alert
    critical = day_data[day_data["Peak_Hour_Prob"] > 0.7]

    if not critical.empty:
        st.error("⚠️ Critical Peak Hours:")
        for _, row in critical.iterrows():
            st.write(row["Datetime"].strftime("%H:%M"))
    else:
        st.success("No critical peak hours")

else:
    st.warning("No hourly probability data available.")


# ============================================
# FEATURE IMPORTANCE
# ============================================

st.header("Feature Importance (Peak Hour Model)")

features = [
    "Load",
    "Temperature",
    "Humidity",
    "Hour",
    "Month",
    "IsWeekend"
]

importance = peak_hour_model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values("Importance", ascending=False)

st.bar_chart(
    importance_df.set_index("Feature")
)
