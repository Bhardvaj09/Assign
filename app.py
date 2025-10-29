
import os
import pandas as pd
import streamlit as st
from langchain_openai import ChatOpenAI

# Streamlit Page Config
st.set_page_config(
    page_title="Chat with CSV - Data Analyst Agent",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Chat with Your CSV Data")

# 🔑 Step 1: API Key Input
st.sidebar.header("🔐 OpenAI API Setup")
api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password")

if not api_key:
    st.sidebar.warning("Please enter your API key to continue.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("💬 **Ask me anything about your data below!**")

# System Prompt
SYSTEM_PROMPT = """You are an expert data analyst and Python programmer.
Analyze the uploaded CSV data and accurately answer questions or generate Pandas code.
Be concise, and only use data provided."""

# 🔹 Step 2: File Upload
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Initialize OpenAI LLM
        try:
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                openai_api_key=api_key
            )

            # Maintain chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Chat input (no title above)
            query = st.text_input(
                "",
                placeholder="Type your question or ask for code... (e.g., 'Show top 5 rows where sales > 5000')"
            )

            ask_button = st.button("🔍 Ask", type="primary")

            if ask_button and query.strip():
                st.session_state.messages.append({"role": "user", "content": query})

                with st.spinner("🤔 Thinking..."):
                    data_summary = f"""
Dataset Overview:
- Shape: {df.shape[0]} rows, {df.shape[1]} columns
- Columns: {', '.join(df.columns.tolist())}
- Data types: {df.dtypes.to_dict()}
"""

                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "system", "content": f"Here is the data description:\n{data_summary}"},
                        *st.session_state.messages
                    ]

                    try:
                        response = llm.invoke(messages)
                        answer = response.content

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer
                        })

                        st.success("✅ Response:")
                        st.write(answer)

                    except Exception as e:
                        st.error(f"⚠️ Error: {str(e)}")
                        st.info("💡 Try rephrasing your question or check your API key.")

            elif ask_button and not query.strip():
                st.warning("⚠️ Please enter a valid question.")

            # Display Chat History
            if st.session_state.messages:
                st.subheader("📜 Conversation History")
                for i, msg in enumerate(st.session_state.messages):
                    if msg["role"] == "user":
                        st.markdown(f"**🙋 You:** {msg['content']}")
                    else:
                        st.markdown(f"**🤖 Assistant:** {msg['content']}")
                    if i < len(st.session_state.messages) - 1:
                        st.divider()

        except Exception as e:
            st.error(f"🔑 Error initializing OpenAI: {str(e)}")

    except Exception as e:
        st.error(f"❌ Error reading CSV file: {str(e)}")

else:
    st.info("👆 Upload a CSV file to begin asking questions.")
