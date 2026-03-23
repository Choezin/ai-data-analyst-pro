import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from PyPDF2 import PdfReader

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

from langchain_openai import ChatOpenAI
from langchain_classic.agents import initialize_agent
from langchain_classic.tools import Tool

# -------------------------------
# 🔹 Page Setup
# -------------------------------
st.set_page_config(page_title="AI Data Analyst Pro", layout="wide")
st.title("🤖 AI Data Analyst Pro (Next-Level)")

# -------------------------------
# 🔹 API Configuration
# -------------------------------
st.sidebar.header("🔑 API Configuration")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# -------------------------------
# 🔹 Data Ingestion
# -------------------------------
def load_data(file, ext):
    if ext == "csv":
        return pd.read_csv(file)
    elif ext in ["xlsx", "xls"]:
        return pd.read_excel(file)
    elif ext == "txt":
        text = file.read().decode("utf-8")
        return pd.DataFrame({"text": text.split("\n")})
    elif ext == "pdf":
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return pd.DataFrame({"text": text.split("\n")})

def load_api(url):
    return pd.DataFrame(requests.get(url).json())

# -------------------------------
# 🔹 Data Handling Agent
# -------------------------------
def data_handling(df):
    df = df.drop_duplicates()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    scaler = StandardScaler()
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df

# -------------------------------
# 🔹 ML Agent
# -------------------------------
def analysis(df, target):
    X = df.drop(columns=[target])
    y = df[target]

    X = X.select_dtypes(include='number')

    if y.nunique() < 10:
        model = RandomForestClassifier()
        problem = "classification"
    else:
        model = RandomForestRegressor()
        problem = "regression"

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    result = {"problem_type": problem}

    if problem == "classification":
        result["accuracy"] = accuracy_score(y_test, preds)
    else:
        result["mse"] = mean_squared_error(y_test, preds)

    result["feature_importance"] = dict(zip(X.columns, model.feature_importances_))

    return result

# -------------------------------
# 🔹 PDF Generator
# -------------------------------
def generate_pdf(insights, decisions):
    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()
    content = [Paragraph("AI Data Report", styles['Title'])]

    for i in insights:
        content.append(Paragraph(f"Insight: {i}", styles['Normal']))
    for d in decisions:
        content.append(Paragraph(f"Decision: {d}", styles['Normal']))

    doc.build(content)
    return "report.pdf"

# -------------------------------
# 🔹 Sidebar Input
# -------------------------------
st.sidebar.header("📥 Data Source")
input_type = st.sidebar.selectbox("Input Type", ["File", "API"])

df = None

if input_type == "File":
    file = st.sidebar.file_uploader("Upload File", type=["csv", "xlsx", "txt", "pdf"])
    if file:
        ext = file.name.split(".")[-1]
        df = load_data(file, ext)

else:
    url = st.sidebar.text_input("Enter API URL")
    if url:
        df = load_api(url)

# -------------------------------
# 🔹 Main App
# -------------------------------
if df is not None:

    st.subheader("📊 Raw Data")
    st.dataframe(df.head())

    if st.button("🧹 Clean Data"):
        df = data_handling(df)
        st.success("Data cleaned successfully!")
        st.dataframe(df.head())

    target = st.selectbox("🎯 Select Target Column", df.columns)

    if st.button("🚀 Run Analysis"):

        result = analysis(df, target)
        st.subheader("🤖 ML Results")
        st.json(result)

        # Insights
        insights = []
        if "accuracy" in result and result["accuracy"] > 0.8:
            insights.append("High accuracy model")

        top_features = sorted(result["feature_importance"].items(),
                              key=lambda x: x[1], reverse=True)[:3]
        insights.append(f"Top features: {top_features}")

        st.subheader("💡 Insights")
        for i in insights:
            st.write("👉", i)

        # Decisions
        decisions = ["Focus on top features", "Deploy model if validated"]

        st.subheader("🎯 Decisions")
        for d in decisions:
            st.write("✅", d)

        # Plotly Visualization
        st.subheader("📊 Advanced Dashboard")
        num_cols = df.select_dtypes(include='number').columns

        if len(num_cols) > 0:
            col = st.selectbox("Select Column", num_cols)
            fig = px.histogram(df, x=col)
            st.plotly_chart(fig)

        # PDF Download
        if st.button("📄 Generate Report"):
            pdf = generate_pdf(insights, decisions)
            with open(pdf, "rb") as f:
                st.download_button("Download Report", f, file_name="report.pdf")

    # -------------------------------
    # 🔹 Chat Agent (LangChain)
    # -------------------------------
    st.subheader("💬 Ask Your Data")

    query = st.text_input("Ask something...")

    if query:
        if not openai_api_key:
            st.error("Please enter your OpenAI API Key in the sidebar.")
        else:
            def summary(_):
                return df.describe().to_string()

            tools = [
                Tool(name="Summary", func=summary, description="Dataset summary")
            ]

            agent = initialize_agent(
                tools,
                ChatOpenAI(temperature=0, api_key=openai_api_key),
                agent="zero-shot-react-description"
            )

            response = agent.run(query)
            st.write("🤖", response)