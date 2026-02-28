"""
Predictive Maintenance Chatbot - Databricks App

A native Databricks app with a chatbot that can query and visualize
predictive maintenance data across 5 industrial asset types.
"""

import json
import logging
import os

import streamlit as st

from model_serving_utils import is_endpoint_supported, query_chat_endpoint
from utils import get_schema_context, parse_llm_data_response, run_sql_query
from visualization import infer_chart_from_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERVING_ENDPOINT = os.getenv("SERVING_ENDPOINT")
DATA_CONTEXT = get_schema_context()

# System prompt for data visualization requests
DATA_SYSTEM_PROMPT = f"""You are a predictive maintenance data assistant. You help users explore and visualize asset health data.

{DATA_CONTEXT}

When the user asks about data, statistics, trends, or visualizations:
1. Generate valid SQL for the Databricks/Unity Catalog tables above. Use the exact table names with the prefix.
2. Return a JSON object with: sql, chart_type, explanation
- sql: The SQL query (required)
- chart_type: One of bar, line, scatter, pie, heatmap
- explanation: Brief description of what the chart shows

Example response for "failure rates by machine type":
```json
{{"sql": "SELECT Type, SUM(\\"Machine failure\") as failures, COUNT(*) as total FROM ... GROUP BY Type", "chart_type": "bar", "explanation": "Failure counts by CNC machine type (H, L, M)"}}
```

Keep queries efficient (use LIMIT when appropriate). For time series use line charts. For categories use bar or pie.
If the user's question is NOT about data/visualization, respond normally in plain text without JSON."""

# General chat system prompt
CHAT_SYSTEM_PROMPT = f"""You are a predictive maintenance assistant. You help users understand:
- CNC machine failures, electrical faults, battery degradation, NASA turbofan RUL, power transformer health
- The datasets cover 5 industrial domains with sensor data, failure flags, and remaining useful life.

{DATA_CONTEXT}

Answer questions about the data, suggest analyses, and help interpret results. Be concise and helpful."""


def run_data_query(user_message: str, chat_history: list) -> tuple[str, object | None]:
    """
    Ask the LLM to generate SQL + chart, execute it, and return response + Plotly figure.
    """
    messages = [
        {"role": "system", "content": DATA_SYSTEM_PROMPT},
        *[{"role": m["role"], "content": m["content"]} for m in chat_history[-6:]],
        {"role": "user", "content": user_message},
    ]

    try:
        response = query_chat_endpoint(SERVING_ENDPOINT, messages, max_tokens=1024)
        parsed = parse_llm_data_response(response)

        if parsed and "sql" in parsed:
            sql_query = parsed["sql"].strip()
            chart_type = parsed.get("chart_type", "bar")
            explanation = parsed.get("explanation", "")

            # Execute SQL
            df = run_sql_query(sql_query)
            fig = infer_chart_from_data(df, chart_type)

            text = f"**{explanation}**\n\nQuery returned {len(df)} rows."
            if not df.empty and len(df) <= 20:
                text += f"\n\n| " + " | ".join(df.columns) + " |\n|" + "|".join(["---"] * len(df.columns)) + "|\n"
                for _, row in df.iterrows():
                    text += "| " + " | ".join(str(v) for v in row) + " |\n"

            return text, fig

        # No structured data response - return raw response as plain chat
        return response, None

    except Exception as e:
        logger.exception("Data query failed")
        return f"Sorry, I couldn't process that: {str(e)}", None


def run_chat(user_message: str, chat_history: list) -> str:
    """General chat without data visualization."""
    messages = [
        {"role": "system", "content": CHAT_SYSTEM_PROMPT},
        *[{"role": m["role"], "content": m["content"]} for m in chat_history[-6:]],
        {"role": "user", "content": user_message},
    ]
    return query_chat_endpoint(SERVING_ENDPOINT, messages, max_tokens=1024)


def main():
    st.set_page_config(
        page_title="Predictive Maintenance Chatbot",
        page_icon="🔧",
        layout="wide",
    )

    st.title("🔧 Predictive Maintenance Chatbot")
    st.markdown(
        "Ask questions about your asset data. I can run SQL and create visualizations for "
        "CNC machines, electrical faults, batteries, turbofan engines, and power transformers."
    )

    if not SERVING_ENDPOINT:
        st.error(
            "**SERVING_ENDPOINT** is not set. Configure a serving endpoint resource in app.yaml "
            "with a foundation model (e.g. databricks-meta-llama-3-3-70b-instruct) or an agent. "
            "See [Databricks Apps docs](https://docs.databricks.com/en/dev-tools/databricks-apps/resources)."
        )
        return

    if "your-endpoint" in (SERVING_ENDPOINT or "").lower() or SERVING_ENDPOINT == "your-endpoint-name":
        st.error(
            "**SERVING_ENDPOINT** is still set to the placeholder. Replace it with your actual endpoint name.\n\n"
            "**To find your endpoint:** Mosaic AI → Model Serving → copy the endpoint name "
            "(e.g. `databricks-meta-llama-3-3-70b-instruct`).\n\n"
            "**To create one:** Model Serving → Create serving endpoint → choose a foundation model "
            "with Chat task type."
        )
        return

    if not is_endpoint_supported(SERVING_ENDPOINT):
        st.error(
            f"The endpoint `{SERVING_ENDPOINT}` is not compatible. "
            "Use a chat-completions endpoint (foundation model or agent). "
            "In Model Serving, ensure the endpoint has task type **Chat** or **Agent**."
        )
        return

    warehouse_id = os.getenv("DATABRICKS_WAREHOUSE_ID")
    if not warehouse_id:
        st.warning(
            "**DATABRICKS_WAREHOUSE_ID** is not set. Data visualization will not work. "
            "Add a SQL warehouse resource in app.yaml."
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "figure" in msg and msg["figure"] is not None:
                st.plotly_chart(msg["figure"], use_container_width=True)

    if prompt := st.chat_input("Ask about failure rates, trends, or asset health..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if warehouse_id:
                    response_text, fig = run_data_query(prompt, st.session_state.messages)
                else:
                    response_text = run_chat(prompt, st.session_state.messages)
                    fig = None

                st.markdown(response_text)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "figure": fig,
        })


if __name__ == "__main__":
    main()
