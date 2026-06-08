
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

st.subheader("Revenue Distribution")
fig = px.histogram(df, x="Revenue")
st.plotly_chart(fig)

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

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor()
model.fit(X_train, y_train)

pred = model.predict(X_test)

score = r2_score(y_test, pred)

st.subheader("Model Accuracy")
st.write(f"R² Score: {score:.2f}")

st.sidebar.header("Scenario Simulation")

aov = st.sidebar.slider("AOV", 10, 200, 50)
orders = st.sidebar.slider("Monthly Orders", 100, 10000, 1000)
instore = st.sidebar.slider("InStore Share", 0.0, 1.0, 0.4)
ue = st.sidebar.slider("Uber Eats Share", 0.0, 1.0, 0.3)
dd = st.sidebar.slider("DoorDash Share", 0.0, 1.0, 0.2)
sd = st.sidebar.slider("Self Delivery Share", 0.0, 1.0, 0.1)
commission = st.sidebar.slider("Commission Rate", 0.0, 0.5, 0.2)

input_df = pd.DataFrame({
    "AOV":[aov],
    "MonthlyOrders":[orders],
    "InStoreShare":[instore],
    "UE_share":[ue],
    "DD_share":[dd],
    "SD_share":[sd],
    "CommissionRate":[commission]
})

prediction = model.predict(input_df)[0]

st.subheader("Predicted Net Profit")
st.success(f"${prediction:,.2f}")
