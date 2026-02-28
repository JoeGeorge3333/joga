"""
Predictive Maintenance Chatbot - Databricks App

A native Databricks app with a chatbot that can query and visualize
data from Databricks sample datasets and predictive maintenance tables.
"""

import logging
import os

import streamlit as st

from model_serving_utils import is_endpoint_supported, query_chat_endpoint
from utils import clean_sql, get_schema_context, parse_llm_data_response, run_sql_query
from visualization import infer_chart_from_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERVING_ENDPOINT = os.getenv("SERVING_ENDPOINT")

# Example prompts for DataKnobs predictive maintenance data
EXAMPLE_PROMPTS = [
    "Show me CNC machine failure rates by type (H, L, M)",
    "Plot failure type distribution (TWF, HDF, PWF, OSF, RNF)",
    "Battery capacity degradation over cycles",
    "Electrical fault counts by phase (A, B, C, Ground)",
    "Transformer oil temperature vs winding temperature over time",
]


def _build_data_prompt() -> str:
    data_context = get_schema_context()
    prefix = os.getenv("DATA_CATALOG", "dataknobs_predictive_maintenance_and_asset_management") + "." + os.getenv("DATA_SCHEMA", "datasets")
    return f"""You are a data assistant for DataKnobs predictive maintenance data. You help users explore and visualize asset health data.

{data_context}

When the user asks about data, statistics, trends, or visualizations:
1. Generate valid SQL for the tables above. Use exact table names with the prefix {prefix}.
2. Return ONLY a JSON object with: sql, chart_type, explanation
- sql: The SQL query (required, use LIMIT 5000 for large tables)
- chart_type: bar, line, scatter, pie, or heatmap
- explanation: Brief description of the chart

Example for "failure rates by machine type":
```json
{{"sql": "SELECT Type, SUM(\\"Machine failure\") as failures, COUNT(*) as total FROM {prefix}.cnc_data_ai_4_i_2020 GROUP BY Type", "chart_type": "bar", "explanation": "Failure counts by CNC machine type (H, L, M)"}}
```

If the question is NOT about data, respond in plain text without JSON."""


def _build_chat_prompt() -> str:
    data_context = get_schema_context()
    return f"""You are a data assistant for DataKnobs predictive maintenance data. Help users explore CNC machines, electrical faults, batteries, turbofan engines, and power transformers.

{data_context}

Answer questions, suggest analyses, and help interpret results. Be concise."""


def run_data_query(user_message: str, chat_history: list) -> tuple[str, object | None]:
    """
    Ask the LLM to generate SQL + chart, execute it, and return response + Plotly figure.
    """
    system_prompt = _build_data_prompt()
    messages = [
        {"role": "system", "content": system_prompt},
        *[{"role": m["role"], "content": m["content"]} for m in chat_history[-6:]],
        {"role": "user", "content": user_message},
    ]

    try:
        response = query_chat_endpoint(SERVING_ENDPOINT, messages, max_tokens=1024)
        parsed = parse_llm_data_response(response)

        if parsed and "sql" in parsed:
            sql_query = clean_sql(parsed["sql"])
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
    system_prompt = _build_chat_prompt()
    messages = [
        {"role": "system", "content": system_prompt},
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
    st.markdown("Ask questions about your DataKnobs asset data. I'll run SQL and create visualizations.")

    # Sidebar: example prompts
    with st.sidebar:
        st.subheader("📊 DataKnobs datasets")
        st.caption("CNC, electrical faults, battery, turbofan, transformer")
        st.divider()
        st.caption("Try these prompts:")
        for ex in EXAMPLE_PROMPTS[:5]:
            if st.button(ex, key=ex[:20], use_container_width=True):
                st.session_state.suggested_prompt = ex
                st.rerun()

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

    # Use suggested prompt from sidebar button
    prompt_input = st.chat_input("Ask about failure rates, asset health, or trends...")
    if "suggested_prompt" in st.session_state:
        prompt_input = st.session_state.pop("suggested_prompt", None)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "figure" in msg and msg["figure"] is not None:
                st.plotly_chart(msg["figure"], use_container_width=True)

    if prompt := prompt_input:
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
