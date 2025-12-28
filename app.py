import pandas as pd
import streamlit as st
from langchain_openai import ChatOpenAI

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="ML Engineer Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Machine Learning Engineer Agent")

# --------------------------------------------------
# SIDEBAR – API KEY
# --------------------------------------------------
st.sidebar.header("🔑 OpenAI API Setup")
api_key = st.sidebar.text_input(
    "Enter your OpenAI API key",
    type="password"
)

if not api_key:
    st.sidebar.warning("Please enter your OpenAI API key to continue.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.info(
    """
This agent is specialized for **Machine Learning only**.

It can:
- Identify ML problem type
- Generate ML code
- Explain models & metrics
- Handle preprocessing & evaluation
"""
)

# --------------------------------------------------
# SYSTEM PROMPT – ML AGENT
# --------------------------------------------------
SYSTEM_PROMPT = """
You are a senior Machine Learning Engineer.

Your responsibilities:
- Work strictly with the uploaded dataset
- Identify whether the problem is regression, classification, or clustering
- Suggest appropriate machine learning models
- Generate correct, clean, and production-ready Python code
- Use pandas, numpy, scikit-learn, matplotlib, and seaborn
- Handle preprocessing: missing values, encoding, scaling, train-test split
- Train models and evaluate them using proper metrics
- Explain model choice, assumptions, and trade-offs
- Explain the code step-by-step clearly

Rules:
- Never assume columns or values not present in the dataset
- Never hallucinate data
- If ML is not suitable for the dataset, clearly say so
- If the user asks for only code, return only code
- If the user asks for explanation, explain in detail
- Keep responses focused on Machine Learning only
"""

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload your CSV dataset",
    type=["csv"]
)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        st.success("✅ Dataset loaded successfully")
        st.write("### Preview of Dataset")
        st.dataframe(df.head())

        # --------------------------------------------------
        # DATA CONTEXT FOR MODEL
        # --------------------------------------------------
        data_summary = f"""
Dataset Overview:
- Rows: {df.shape[0]}
- Columns: {df.shape[1]}

Column Names:
{', '.join(df.columns)}

Data Types:
{df.dtypes.to_string()}

Sample Rows:
{df.head(5).to_string(index=False)}

Summary Statistics:
{df.describe(include='all').to_string()}
"""

        # --------------------------------------------------
        # LLM INIT
        # --------------------------------------------------
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=api_key
        )

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # --------------------------------------------------
        # USER QUERY
        # --------------------------------------------------
        query = st.text_input(
            "",
            placeholder=(
                "Ask ML questions like:\n"
                "• Build a regression model to predict sales\n"
                "• Which model is best and why?\n"
                "• Give code for classification with evaluation\n"
                "• Explain preprocessing steps"
            )
        )

        ask_button = st.button("🚀 Generate", type="primary")

        if ask_button and query.strip():
            st.session_state.messages.append(
                {"role": "user", "content": query}
            )

            with st.spinner("🤖 Thinking like an ML Engineer..."):
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "system", "content": "Use ONLY the dataset provided below."},
                    {"role": "system", "content": data_summary},
                    *st.session_state.messages
                ]

                try:
                    response = llm.invoke(messages)
                    answer = response.content

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )

                    st.success("✅ Response")
                    st.markdown(answer)

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

        elif ask_button and not query.strip():
            st.warning("⚠️ Please enter a valid ML question.")

    except Exception as e:
        st.error(f"❌ Error reading CSV file: {str(e)}")

else:
    st.info("👆 Upload a CSV file to start using the ML Agent")
