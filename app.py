import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# --- Load environment variables ---
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Chat with CSV - Data Analyst Agent",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Chat with Your CSV Data")

# --- API Key Section ---
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        api_key = None

# Allow user to manually enter API key if not found
if not api_key:
    api_key = st.text_input("🔑 Enter your OpenAI API Key:", type="password")
    if not api_key:
        st.warning("Please provide your OpenAI API key to continue.")
        st.stop()

# --- System Prompt ---
SYSTEM_PROMPT = """You are an expert data analyst. Analyze the data and answer questions accurately.
When asked about data, provide clear, concise answers based only on the information available.
If asked to perform calculations, show your work."""

# --- File Upload Section ---
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file:
    try:
        # Read CSV file
        df = pd.read_csv(uploaded_file)

        # Display data preview
        st.subheader("📄 Data Preview")
        st.dataframe(df.head(10))

        # Display dataset info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB")

        # Show column info
        with st.expander("📋 Column Information"):
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.values,
                'Non-Null Count': df.count().values,
                'Null Count': df.isnull().sum().values
            })
            st.dataframe(col_info)

        # --- Initialize LLM ---
        try:
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                openai_api_key=api_key
            )

            if "messages" not in st.session_state:
                st.session_state.messages = []

            st.subheader("💬 Ask Questions About Your Data")

            # Query input
            query = st.text_input(
                "Type your question here:",
                placeholder="e.g., What is the average value in the sales column?"
            )

            # Buttons
            col1, col2 = st.columns([1, 5])
            with col1:
                ask_button = st.button("🔍 Ask", type="primary")
            with col2:
                clear_button = st.button("🗑️ Clear History")

            if clear_button:
                st.session_state.messages = []
                st.success("Chat history cleared!")
                st.rerun()

            if ask_button and query.strip():
                st.session_state.messages.append({"role": "user", "content": query})
                with st.spinner("🤔 Analyzing your data..."):
                    try:
                        data_summary = f"""
Dataset Information:
- Shape: {df.shape[0]} rows, {df.shape[1]} columns
- Columns: {', '.join(df.columns.tolist())}
- Data types: {df.dtypes.to_dict()}

First few rows:
{df.head(10).to_string()}

Summary statistics:
{df.describe().to_string()}
"""
                        messages = [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "system", "content": f"Here's the data you're analyzing:\n{data_summary}"},
                            {"role": "user", "content": query}
                        ]
                        response = llm.invoke(messages)
                        answer = response.content
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        st.success("✅ Answer:")
                        st.write(answer)
                    except Exception as e:
                        st.error(f"⚠️ Error: {str(e)}")
                        st.info("💡 Try rephrasing your question or check your API key.")
            elif ask_button and not query.strip():
                st.warning("⚠️ Please enter a question.")

            # Display chat history
            if st.session_state.messages:
                st.subheader("📜 Chat History")
                for i, msg in enumerate(st.session_state.messages):
                    if msg["role"] == "user":
                        st.markdown(f"**🙋 You:** {msg['content']}")
                    else:
                        st.markdown(f"**🤖 Assistant:** {msg['content']}")
                    if i < len(st.session_state.messages) - 1:
                        st.divider()

            # --- Visualization Section ---
            st.subheader("📊 Create Custom Visualizations")

            with st.expander("🎨 Chart Builder"):
                chart_col1, chart_col2 = st.columns(2)
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                all_cols = df.columns.tolist()

                with chart_col1:
                    x_axis = st.selectbox("Select X-axis", all_cols)
                    chart_type = st.selectbox(
                        "Select Chart Type",
                        ["Bar", "Line", "Scatter", "Pie", "Histogram", "Box Plot"]
                    )

                with chart_col2:
                    y_axis = st.selectbox("Select Y-axis", numeric_cols if numeric_cols else all_cols)

                if st.button("📈 Generate Chart"):
                    fig, ax = plt.subplots(figsize=(10, 6))
                    try:
                        if chart_type == "Bar":
                            data = df.groupby(x_axis)[y_axis].sum().sort_values(ascending=False).head(10)
                            data.plot(kind="bar", ax=ax, color='steelblue')
                            ax.set_title(f"Top 10 {y_axis} by {x_axis}")
                            plt.xticks(rotation=45, ha='right')

                        elif chart_type == "Line":
                            data = df.groupby(x_axis)[y_axis].sum()
                            data.plot(kind="line", ax=ax, marker='o', linewidth=2, color='steelblue')

                        elif chart_type == "Scatter":
                            ax.scatter(df[x_axis], df[y_axis], alpha=0.6, color='steelblue', s=50)

                        elif chart_type == "Pie":
                            data = df.groupby(x_axis)[y_axis].sum().head(10)
                            data.plot(kind="pie", ax=ax, autopct='%1.1f%%', startangle=90)
                            ax.set_ylabel('')

                        elif chart_type == "Histogram":
                            df[y_axis].hist(ax=ax, bins=20, color='steelblue', edgecolor='black')

                        elif chart_type == "Box Plot":
                            df.boxplot(column=y_axis, by=x_axis, ax=ax)
                            plt.suptitle('')

                        plt.tight_layout()
                        st.pyplot(fig)

                    except Exception as e:
                        st.error(f"Could not generate chart: {str(e)}")
                        st.info("💡 Try selecting different columns or chart type.")

        except Exception as init_error:
            st.error(f"🔑 Error initializing OpenAI: {str(init_error)}")
            st.info("Please check that your API key is valid and has not expired.")

    except Exception as e:
        st.error(f"❌ Error reading CSV file: {str(e)}")
        st.info("Please make sure your CSV file is properly formatted.")
else:
    st.info("👆 Upload a CSV file to get started!")
