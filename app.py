import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Page configuration
st.set_page_config(
    page_title="Chat with CSV - Data Analyst Agent",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Chat with Your CSV Data")
st.markdown("Upload a CSV file and ask questions about your data using natural language!")

# System prompt
SYSTEM_PROMPT = """You are an expert data analyst. Analyze the data and answer questions accurately.
When asked about data, provide clear, concise answers based only on the information available.
If asked to perform calculations, show your work."""

# File uploader
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
        
        # Show column information
        with st.expander("📋 Column Information"):
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.values,
                'Non-Null Count': df.count().values,
                'Null Count': df.isnull().sum().values
            })
            st.dataframe(col_info)
        
        # Initialize LLM
        if api_key:
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0,
                api_key=api_key
            )
            
            # Initialize session state
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            # Chat interface
            st.subheader("💬 Ask Questions About Your Data")
            
            # Example questions
            with st.expander("💡 Example Questions"):
                st.markdown("""
                - What are the column names?
                - Show me summary statistics
                - What is the average of [column_name]?
                - How many unique values are in [column_name]?
                - What are the top 5 rows sorted by [column_name]?
                - Are there any missing values?
                - What's the correlation between [column1] and [column2]?
                """)
            
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
            
            # Process query
            if ask_button and query.strip():
                # Add user message to history
                st.session_state.messages.append({
                    "role": "user",
                    "content": query
                })
                
                with st.spinner("🤔 Analyzing your data..."):
                    try:
                        # Prepare data context
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
                        
                        # Create messages for LLM
                        messages = [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "system", "content": f"Here's the data you're analyzing:\n{data_summary}"},
                            {"role": "user", "content": query}
                        ]
                        
                        # Get response from LLM
                        response = llm.invoke(messages)
                        answer = response.content
                        
                        # Add assistant response to history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer
                        })
                        
                        # Display response
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
            
            # Manual chart creation section
            st.subheader("📊 Create Custom Visualizations")
            
            with st.expander("🎨 Chart Builder"):
                chart_col1, chart_col2 = st.columns(2)
                
                # Get numeric and categorical columns
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
                            ax.set_title(f"Top 10 {y_axis} by {x_axis}", fontsize=14, fontweight='bold')
                            ax.set_xlabel(x_axis)
                            ax.set_ylabel(y_axis)
                            plt.xticks(rotation=45, ha='right')
                            
                        elif chart_type == "Line":
                            data = df.groupby(x_axis)[y_axis].sum()
                            data.plot(kind="line", ax=ax, marker='o', linewidth=2, color='steelblue')
                            ax.set_title(f"{y_axis} Trend by {x_axis}", fontsize=14, fontweight='bold')
                            ax.set_xlabel(x_axis)
                            ax.set_ylabel(y_axis)
                            ax.grid(True, alpha=0.3)
                            
                        elif chart_type == "Scatter":
                            ax.scatter(df[x_axis], df[y_axis], alpha=0.6, color='steelblue', s=50)
                            ax.set_xlabel(x_axis)
                            ax.set_ylabel(y_axis)
                            ax.set_title(f"{y_axis} vs {x_axis}", fontsize=14, fontweight='bold')
                            ax.grid(True, alpha=0.3)
                            
                        elif chart_type == "Pie":
                            data = df.groupby(x_axis)[y_axis].sum().head(10)
                            data.plot(kind="pie", ax=ax, autopct='%1.1f%%', startangle=90)
                            ax.set_title(f"{y_axis} Distribution by {x_axis}", fontsize=14, fontweight='bold')
                            ax.set_ylabel('')
                            
                        elif chart_type == "Histogram":
                            df[y_axis].hist(ax=ax, bins=20, color='steelblue', edgecolor='black')
                            ax.set_title(f"Distribution of {y_axis}", fontsize=14, fontweight='bold')
                            ax.set_xlabel(y_axis)
                            ax.set_ylabel('Frequency')
                            ax.grid(True, alpha=0.3)
                            
                        elif chart_type == "Box Plot":
                            df.boxplot(column=y_axis, by=x_axis, ax=ax)
                            ax.set_title(f"{y_axis} Distribution by {x_axis}", fontsize=14, fontweight='bold')
                            ax.set_xlabel(x_axis)
                            ax.set_ylabel(y_axis)
                            plt.suptitle('')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f"Could not generate chart: {str(e)}")
                        st.info("💡 Try selecting different columns or chart type.")
        
        else:
            st.error("🔑 OpenAI API key not found!")
            st.info("Please add your API key in Streamlit Cloud secrets or create a `.env` file with `OPENAI_API_KEY=your_key_here`")
            
            # Show how to add secrets in Streamlit Cloud
            with st.expander("How to add API key in Streamlit Cloud"):
                st.code("""
1. Go to your app settings in Streamlit Cloud
2. Click on "Secrets"
3. Add:
OPENAI_API_KEY = "your-api-key-here"
                """)
    
    except Exception as e:
        st.error(f"❌ Error reading CSV file: {str(e)}")
        st.info("Please make sure your CSV file is properly formatted.")

else:
    # Welcome message when no file is uploaded
    st.info("👆 Upload a CSV file to get started!")
    
    st.markdown("""
    ### 🚀 How to Use:
    1. Upload your CSV file using the uploader above
    2. View the data preview and column information
    3. Ask questions in natural language
    4. Create custom visualizations
    
    ### 🔧 Setup:
    For local development, create a `.env` file:
    ```
    OPENAI_API_KEY=your_openai_api_key_here
    ```
    
    For Streamlit Cloud, add your API key in the app secrets.
    """)
