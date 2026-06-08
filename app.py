import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

st.set_page_config(page_title="Restaurant Profit Optimization", layout="wide")

df = pd.read_csv("restaurant_data.csv")

st.title("Predictive Modeling & Profit Optimization Dashboard")

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Net Profit Distribution")

fig = px.histogram(
    df,
    x="NetProfit",
    nbins=30,
    title="Net Profit Distribution"
)

fig.update_layout(
    xaxis_title="Net Profit",
    yaxis_title="Count",
    bargap=0.1
)

st.plotly_chart(fig, use_container_width=True)

features = [
    "AOV",
    "MonthlyOrders",
    "InStoreShare",
    "UE_share",
    "DD_share",
    "SD_share",
    "CommissionRate"
]

target = "NetProfit"

features = [col for col in features if col in df.columns]

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor()

model.fit(X_train, y_train)

pred = model.predict(X_test)

score = r2_score(y_test, pred)

st.subheader("Model Accuracy")
st.success(f"R² Score: {score:.2f}")

st.sidebar.header("Scenario Simulation")

input_data = {}

for col in features:

    min_val = float(df[col].min())
    max_val = float(df[col].max())
    mean_val = float(df[col].mean())

    input_data[col] = st.sidebar.slider(
        col,
        min_value=min_val,
        max_value=max_val,
        value=mean_val
    )

input_df = pd.DataFrame([input_data])

prediction = model.predict(input_df)[0]

st.subheader("Predicted Net Profit")

st.success(f"${prediction:,.2f}")
