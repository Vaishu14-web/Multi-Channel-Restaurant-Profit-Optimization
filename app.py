import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Restaurant Profit Dashboard",
    layout="wide"
)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("restaurant_data.csv")
df.columns = df.columns.str.strip()

st.title("🍽️ Restaurant Profit Optimization Dashboard")
st.markdown("### Power BI Style AI Analytics System")

st.markdown("---")

# ---------------- SAFE NUMERIC HANDLING ----------------
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

# Safety check
if len(numeric_cols) < 2:
    st.error("Dataset must contain at least 2 numeric columns (features + target).")
    st.stop()

# FIX: safer target selection (last numeric column)
target = numeric_cols[-1]
features = numeric_cols[:-1]

X = df[features]
y = df[target]

# ---------------- MODEL TRAINING ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
score = r2_score(y_test, pred)

# ---------------- KPI DASHBOARD ----------------
st.subheader("📊 Business Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Records", df.shape[0])
col2.metric("Features", len(features))
col3.metric("Model Accuracy (R²)", f"{score:.2f}")
col4.metric("Target Column", target)

st.markdown("---")

# ---------------- DATA PREVIEW ----------------
st.subheader("📁 Dataset Preview")
st.dataframe(df.head())

# ---------------- SAFE CORRELATION HEATMAP ----------------
st.subheader("📈 Business Intelligence Insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Correlation Heatmap")

    if len(numeric_cols) > 1:
        fig, ax = plt.subplots()
        sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Not enough numeric columns for heatmap")

# ---------------- FEATURE IMPORTANCE ----------------
with col2:
    st.markdown("### Feature Importance")

    importances = model.feature_importances_

    fig, ax = plt.subplots()
    ax.barh(features, importances)
    ax.set_title("Impact on Target")
    st.pyplot(fig)

st.markdown("---")

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("⚙️ Scenario Simulation")

input_data = {}

for col in features:
    min_val = float(df[col].min())
    max_val = float(df[col].max())
    mean_val = float(df[col].mean())

    # Safety fix: handle constant columns
    if min_val == max_val:
        max_val = min_val + 1

    input_data[col] = st.sidebar.slider(
        col,
        min_value=min_val,
        max_value=max_val,
        value=mean_val
    )

input_df = pd.DataFrame([input_data])

# ---------------- PREDICTION ----------------
prediction = model.predict(input_df)[0]

st.subheader("🤖 Prediction Engine")

col1, col2 = st.columns(2)

with col1:
    st.info("Adjust values in sidebar to simulate business scenario")

with col2:
    st.success(f"💰 Predicted Value: {prediction:,.2f}")
