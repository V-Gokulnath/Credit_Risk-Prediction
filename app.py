import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load model
model = joblib.load("model.pkl")

# Page configuration
st.set_page_config(page_title="Credit Risk Prediction", layout="wide")

# Title
st.title("💳 Credit Risk Prediction Dashboard")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("credit_risk_dataset.csv")

df = load_data()

# -------- Helper function for prediction --------
def prepare_input(df, age, income, loan_amount, loan_int_rate):
    input_df = df.iloc[[0]].copy()
    input_df["person_age"] = age
    input_df["person_income"] = income
    input_df["loan_amnt"] = loan_amount
    input_df["loan_int_rate"] = loan_int_rate
    return input_df

# Sidebar
st.sidebar.header("Menu")
option = st.sidebar.selectbox(
    "Select View",
    ["Dataset Overview", "EDA Visualizations", "Risk Analysis", "Predict Credit Risk"]
)

# ---------------- DATASET OVERVIEW ----------------
if option == "Dataset Overview":
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Information")
    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")

    st.subheader("Column Names")
    st.write(list(df.columns))

# ---------------- EDA VISUALIZATIONS ----------------
elif option == "EDA Visualizations":
    st.subheader("Exploratory Data Analysis")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    feature = st.selectbox("Select Feature", numeric_cols)

    fig, ax = plt.subplots()
    sns.histplot(df[feature], kde=True, ax=ax)
    st.pyplot(fig)

# ---------------- RISK ANALYSIS ----------------
elif option == "Risk Analysis":
    st.subheader("Credit Risk Distribution")

    target_col = "loan_status"  # change if needed

    if target_col in df.columns:
        fig, ax = plt.subplots()
        sns.countplot(x=target_col, data=df, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Target column not found. Update target_col name.")

# ---------------- PREDICTION ----------------
elif option == "Predict Credit Risk":
    st.subheader("🔮 Predict Credit Risk")
    st.write("Enter applicant details below:")

    age = st.number_input("Person Age", min_value=18, max_value=100)
    income = st.number_input("Person Income")
    loan_amount = st.number_input("Loan Amount")
    loan_int_rate = st.number_input("Loan Interest Rate")

    if st.button("Predict"):
        input_df = prepare_input(df, age, income, loan_amount, loan_int_rate)
        prediction = model.predict(input_df)

        if prediction[0] == 1:
            st.error("❌ High Credit Risk")
        else:
            st.success("✅ Low Credit Risk")
