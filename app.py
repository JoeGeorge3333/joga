"""
Predictive Maintenance Chatbot - Databricks App

A native Databricks app with a chatbot that can query and visualize
data from Databricks sample datasets and predictive maintenance tables.
"""

import logging
import os

import pandas as pd
import streamlit as st

from model_serving_utils import is_endpoint_supported, query_chat_endpoint
from utils import clean_sql, get_schema_context, parse_llm_data_response, run_sql_query
from visualization import create_chart_with_selection, infer_chart_from_data

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

When users ask vaguely (e.g. "visualize cnc", "show me data", "exploratory analysis"), suggest specific prompts they can try:
- "Show me CNC machine failure rates by type (H, L, M)"
- "Plot failure type distribution (TWF, HDF, PWF, OSF, RNF)"
- "Battery capacity degradation over cycles"
- "Electrical fault counts by phase"
- "Transformer oil temperature vs winding temperature over time"

Answer questions, suggest analyses, and help interpret results. Be concise."""


def run_data_query(user_message: str, chat_history: list) -> tuple[str, object | None, "pd.DataFrame | None", str]:
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
    except Exception as e:
        logger.exception("Model endpoint request failed")
        err = str(e)
        hint = "Check that the model serving endpoint is running and the app has Can Query permission."
        return f"**Model request failed:** {err}\n\n{hint}", None, None, "bar"

    parsed = parse_llm_data_response(response)

    if not parsed or "sql" not in parsed:
        return response, None, None, "bar"

    sql_query = clean_sql(parsed["sql"])
    chart_type = parsed.get("chart_type", "bar")
    explanation = parsed.get("explanation", "")

    try:
        df = run_sql_query(sql_query)
    except Exception as e:
        logger.exception("SQL execution failed")
        err = str(e)
        hint = "Check: (1) DataKnobs tables exist in Unity Catalog, (2) SQL warehouse is running, (3) App has access to the catalog."
        return f"**SQL execution failed:** {err}\n\n**Query attempted:**\n```sql\n{sql_query}\n```\n\n{hint}", None, None, "bar"

    try:
        fig = infer_chart_from_data(df, chart_type)
    except Exception as e:
        logger.exception("Chart generation failed")
        return f"**Chart failed:** {e}\n\nQuery returned {len(df)} rows. Data preview:\n{table_preview(df)}", None, df, chart_type

    text = f"**{explanation}**\n\nQuery returned {len(df)} rows."
    if not df.empty and len(df) <= 20:
        text += f"\n\n{table_preview(df)}"

    return text, fig, df, chart_type


def table_preview(df) -> str:
    """Format a small DataFrame as markdown table."""
    if df.empty or len(df) > 50:
        return ""
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def run_chat(user_message: str, chat_history: list) -> str:
    """General chat without data visualization."""
    system_prompt = _build_chat_prompt()
    messages = [
        {"role": "system", "content": system_prompt},
        *[{"role": m["role"], "content": m["content"]} for m in chat_history[-6:]],
        {"role": "user", "content": user_message},
    ]
    try:
        return query_chat_endpoint(SERVING_ENDPOINT, messages, max_tokens=1024)
    except Exception as e:
        logger.exception("Model endpoint request failed")
        return f"**Model request failed:** {str(e)}\n\nCheck that the model serving endpoint is running and the app has Can Query permission."


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
    if "viz_data" not in st.session_state:
        st.session_state.viz_data = None
    if "viz_chart_type" not in st.session_state:
        st.session_state.viz_chart_type = "bar"

    # Use suggested prompt from sidebar button
    prompt_input = st.chat_input("Ask about failure rates, asset health, or trends...")
    if "suggested_prompt" in st.session_state:
        prompt_input = st.session_state.pop("suggested_prompt", None)

    # Two-column layout: chat left, interactive viz right
    chat_col, viz_col = st.columns([3, 2], gap="large")

    with chat_col:
        st.subheader("💬 Chat")
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
                        response_text, fig, df, chart_type = run_data_query(prompt, st.session_state.messages)
                        if df is not None and fig is not None:
                            st.session_state.viz_data = df
                            st.session_state.viz_chart_type = chart_type
                    else:
                        response_text = run_chat(prompt, st.session_state.messages)
                        fig = None
                        df = None

                    st.markdown(response_text)
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "figure": fig,
            })
            st.rerun()

    with viz_col:
        st.subheader("📊 Exploratory Data Analysis")
        st.caption("Visuals update when you ask data questions. Use controls to explore.")

        df = st.session_state.viz_data
        if df is not None and not df.empty:
            cols = list(df.columns)
            numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
            categorical = [c for c in cols if c not in numeric]

            chart_opts = ["bar", "line", "scatter", "pie", "heatmap"]
            default_idx = chart_opts.index(st.session_state.viz_chart_type) if st.session_state.viz_chart_type in chart_opts else 0
            chart_type = st.selectbox(
                "Chart type",
                options=chart_opts,
                index=default_idx,
                key="viz_chart_type_select",
            )

            x_opts = ["(auto)"] + cols
            y_opts = ["(auto)"] + cols
            color_opts = ["(none)"] + cols

            x_col = st.selectbox("X axis", x_opts, key="viz_x")
            y_col = st.selectbox("Y axis", y_opts, key="viz_y")
            color_col = st.selectbox("Color by", color_opts, key="viz_color")

            x_val = None if x_col == "(auto)" else x_col
            y_val = None if y_col == "(auto)" else y_col
            color_val = None if color_col == "(none)" else color_col

            try:
                viz_fig = create_chart_with_selection(
                    df, chart_type,
                    x_col=x_val, y_col=y_val, color_col=color_val,
                    title=f"Data: {len(df)} rows",
                )
                st.plotly_chart(viz_fig, use_container_width=True, key="viz_chart")
            except Exception as e:
                st.error(f"Chart error: {e}")
                viz_fig = infer_chart_from_data(df, chart_type)
                st.plotly_chart(viz_fig, use_container_width=True, key="viz_chart_fallback")

            with st.expander("📋 Data table", expanded=False):
                st.dataframe(df.head(100), use_container_width=True, height=300)

            if st.button("Clear visualization", key="viz_clear"):
                st.session_state.viz_data = None
                st.rerun()
        else:
            st.info("Ask a data question to see visualizations here. Try: *Show me CNC machine failure rates by type*")
            st.markdown("**Example prompts:**")
            for ex in EXAMPLE_PROMPTS[:3]:
                st.markdown(f"- {ex}")


if __name__ == "__main__":
    main()
