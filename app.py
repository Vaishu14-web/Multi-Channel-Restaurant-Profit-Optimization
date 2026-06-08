import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

st.set_page_config(page_title="Restaurant Profit Optimization", layout="wide")

df = pd.read_csv("restaurant_data.csv")

df.columns = df.columns.str.strip()

st.title("Restaurant Profit Optimization Dashboard")

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Dataset Columns")
st.write(df.columns.tolist())

numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

st.subheader("Numeric Columns")
st.write(numeric_cols)

target = numeric_cols[-1]

features = numeric_cols[:-1]

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

st.subheader("Predicted Value")
st.success(f"{prediction:,.2f}")
