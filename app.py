import streamlit as st
import pandas as pd
import plotly.express as px
import json
import numpy as np
from langchain.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub

# # Configure the LLM model for Llama3 via Ollama (as in the backup code)
# def configure_ollama_llama():
#     return OllamaLLM(
#         model="llama3.2",  # Use the llama3.2 model
#         base_url="http://localhost:11434"  # Ollama's default API endpoint
#     )

# llama_model = configure_ollama_llama()

# Replace Ollama configuration with HuggingFaceHub
llama_model = HuggingFaceHub(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    huggingfacehub_api_token="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"
)

st.title("Interactive Dashboard with Llama3 and LangChain Insights")

def is_time_column(series):
    try:
        parsed_dates = pd.to_datetime(series, errors='coerce')
        return parsed_dates.notna().mean() > 0.5
    except Exception:
        return False

def clean_data(df):
    df = df.dropna(how="all", axis=1)
    df = df.dropna(how="all", axis=0)
    if df.columns.duplicated().any():
        st.warning("Duplicate column names detected. Renaming duplicates.")
        df.columns = pd.io.parsers.ParserBase({'names': df.columns})._maybe_dedup_names(df.columns)
    return df

def classify_columns(df):
    categorical = [col for col in df.columns if df[col].dtype == "object"]
    numerical = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    time_based = [col for col in df.columns if is_time_column(df[col])]
    return categorical, numerical, time_based

def create_langchain_rag_tool(data):
    # Convert each row into a Document
    documents = [
        Document(page_content=" ".join(row.astype(str)), metadata={"index": idx})
        for idx, row in data.iterrows()
    ]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever()

    # Use default "stuff" chain without any custom prompts
    qa = RetrievalQA.from_chain_type(
        llm=llama_model,
        retriever=retriever,
        chain_type="stuff"
    )
    return qa

def dict_to_plotly(data, chart_dict):
    st.subheader("Charts from Dictionary")
    charts = chart_dict.get("charts", [])
    justifications = chart_dict.get("justifications", [])

    if not charts:
        st.warning("No valid charts found in the structured data.")
        return

    categorical_columns = [col for col in data.columns if data[col].dtype == 'object']
    numerical_columns = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]

    for idx, chart_info in enumerate(charts):
        chart_type = chart_info.get("type", "").lower()
        title = chart_info.get("title", "Unnamed Chart")
        axes = chart_info.get("axes", [])
        justification = justifications[idx] if idx < len(justifications) else "No justification provided."

        with st.expander(f"{title}"):
            st.write(f"**Justification:** {justification}")
            
            if len(axes) < 2:
                x_axis = axes[0] if axes else (numerical_columns[0] if numerical_columns else None)
                y_axis = (numerical_columns[1] if len(numerical_columns) > 1 else 
                          numerical_columns[0] if numerical_columns else None)
                if x_axis is None or y_axis is None:
                    st.warning(f"Could not generate chart for {title}: Insufficient data columns.")
                    continue
            else:
                x_axis, y_axis = axes[:2]

            if x_axis not in data.columns or y_axis not in data.columns:
                st.warning(f"Columns {x_axis} or {y_axis} not found. Using default columns.")
                x_axis = numerical_columns[0]
                y_axis = numerical_columns[1] if len(numerical_columns) > 1 else numerical_columns[0]

            try:
                if chart_type == "scatter":
                    fig = px.scatter(data, x=x_axis, y=y_axis, title=title)
                elif chart_type == "bar":
                    fig = px.bar(data, x=x_axis, y=y_axis, title=title)
                elif chart_type == "boxplot":
                    fig = px.box(data, x=x_axis, y=y_axis, title=title)
                else:
                    st.warning(f"Unknown chart type {chart_type}. Defaulting to scatter plot.")
                    fig = px.scatter(data, x=x_axis, y=y_axis, title=title)
                
                st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"Error generating chart: {e}")
                st.warning(f"Could not create chart for {title} due to data constraints.")

def generate_human_response(df):
    st.subheader("Llama3 Textual Insights")
    try:
        # A simple human query, no strict instructions
        human_query = """
        Analyze this dataset in detail by looking at its rows and columns.
        Provide a textual summary of key patterns, trends, and correlations.
        Suggest meaningful observations based on this data.
        """
        st.write("### Generating Textual Insights...")
        qa_tool = create_langchain_rag_tool(df)
        human_response = qa_tool.run(human_query)

        if not human_response or len(human_response.strip()) == 0:
            raise ValueError("The LLM returned an empty response for textual insights.")

        # This will print just what the model returns, similar to the backup scenario
        st.write("### Human-readable Insights:")
        st.text(human_response)
        return human_response
    except Exception as e:
        st.error(f"An error occurred while generating textual insights: {e}")
        return None

def generate_structured_response(df, human_response):
    st.subheader("Llama3 Structured Insights")
    try:
        if not human_response or len(human_response.strip()) == 0:
            raise ValueError("The human response is empty or invalid.")

        json_query = f"""
        Based on the following analysis:
        {human_response}

        Generate structured insights in JSON format with this EXACT structure:
        {{
            "charts": [
                {{
                    "type": "scatter|bar|boxplot",
                    "title": "Chart Title",
                    "axes": ["column_name_x", "column_name_y"]
                }}
            ],
            "justifications": ["Explanation for each chart"]
        }}

        Ensure the JSON is complete and valid without any outside text.
        Focus on creating meaningful visualizations that reveal insights from the data.
        """
        st.write("### Generating Structured Insights...")
        qa_tool = create_langchain_rag_tool(df)
        structured_response = qa_tool.run(json_query)
        print(f"Raw JSON response: {structured_response}")

        if not structured_response or len(structured_response.strip()) == 0:
            raise ValueError("The LLM did not return any structured response.")

        try:
            import re
            json_match = re.search(r'\{.*\}', structured_response, re.DOTALL)
            if json_match:
                structured_response = json_match.group(0)
            
            structured_data = json.loads(structured_response)
            print(f"Structured data type: {type(structured_data)}")
            st.write("### Structured Insights (for Charts):")
            st.json(structured_data)
            dict_to_plotly(df, structured_data)
            return structured_data
        except json.JSONDecodeError as e:
            print("Failed to parse JSON response:", e)
            st.error("The AI's response was not valid JSON. Please try again.")
            st.write("### Raw AI Response (Debugging):")
            st.code(structured_response)
            return None
    except Exception as e:
        print(f"Error during structured insights generation: {e}")
        st.error(f"An error occurred while generating structured insights: {e}")
        return None

def create_interactive_dashboard(data):
    st.subheader("Interactive Dashboard (Manual)")
    categorical, numerical, time_based = classify_columns(data)

    use_time_based = st.checkbox("Click this Checkbox if you have a Time-Based Columns.")

    if use_time_based:
        if time_based:
            st.write("### Time-Based Columns:")
            selected_time_column = st.selectbox("Select a Time-Based Column", time_based)
            if numerical:
                y_axis = st.selectbox("Select a Numerical Column for Y-Axis", numerical)
                line_chart = px.line(
                    data,
                    x=selected_time_column,
                    y=y_axis,
                    title=f"Line Chart: {y_axis} over {selected_time_column}",
                )
                st.plotly_chart(line_chart, use_container_width=True)
            else:
                st.warning("No numerical columns available for Y-Axis.")
        else:
            st.warning("No time-based columns found in the dataset.")

    if categorical and numerical:
        st.write("### Categorical and Numerical Data")
        selected_category = st.selectbox("Filter by Categorical Column", categorical)
        selected_category_value = st.selectbox(
            f"Filter Values for {selected_category}",
            options=["All"] + list(data[selected_category].unique()),
        )
        filtered_data = (
            data if selected_category_value == "All"
            else data[data[selected_category] == selected_category_value]
        )

        col1, col2 = st.columns(2)

        with col1:
            bar_chart = px.bar(
                filtered_data,
                x=categorical[0],
                y=numerical[0],
                title=f"Bar Chart: {numerical[0]} by {categorical[0]}",
                color=categorical[0],
            )
            st.plotly_chart(bar_chart, use_container_width=True)

        with col2:
            if len(numerical) > 1:
                scatter_chart = px.scatter(
                    filtered_data,
                    x=numerical[0],
                    y=numerical[1],
                    title=f"Scatter Plot: {numerical[0]} vs {numerical[1]}",
                    color=categorical[0],
                )
                st.plotly_chart(scatter_chart, use_container_width=True)
            else:
                st.warning("Not enough numerical columns for a scatter plot.")
    else:
        st.warning("No sufficient categorical or numerical data available for visualization.")

def main():
    uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            data = clean_data(data)
            dashboard_option = st.sidebar.radio(
                "Select Dashboard",
                ("Interactive Dashboard", "Llama3 Textual Insights", "Llama3 Charts")
            )
            if dashboard_option == "Interactive Dashboard":
                create_interactive_dashboard(data)
            elif dashboard_option == "Llama3 Textual Insights":
                human_response = generate_human_response(data)
                if human_response:
                    structured_response = generate_structured_response(data, human_response)
                    if structured_response:
                        st.session_state["structured_insights"] = structured_response
            elif dashboard_option == "Llama3 Charts":
                if "structured_insights" in st.session_state:
                    dict_to_plotly(data, st.session_state["structured_insights"])
                else:
                    st.warning("No structured insights found. Please generate textual insights first.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
