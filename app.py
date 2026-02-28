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
from utils import (
    clean_sql,
    get_dynamic_schema_context,
    get_schema_context,
    parse_llm_data_response,
    run_sql_query,
)
from visualization import (
    create_chart_with_selection,
    create_correlation_heatmap,
    infer_chart_from_data,
)

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
    "Correlation matrix for CNC numeric columns",
]


def _build_data_prompt(dynamic_schema: str | None = None) -> str:
    data_context = get_schema_context(dynamic_context=dynamic_schema)
    prefix = os.getenv("DATA_CATALOG", "dataknobs_predictive_maintenance_and_asset_management") + "." + os.getenv("DATA_SCHEMA", "datasets")
    return f"""You are a data assistant for DataKnobs predictive maintenance data. You help users explore and visualize asset health data.

{data_context}

When the user asks about data, statistics, trends, or visualizations:
1. Generate valid SQL for the tables above. Use exact table names with the prefix {prefix}.
2. Return ONLY a JSON object with: sql, chart_type, explanation
- sql: The SQL query (required, use LIMIT 5000 for large tables)
- chart_type: bar, line, scatter, pie, heatmap, or correlation (for correlation matrix)
- explanation: Brief description of the chart

**SQL rules:** Quote column names with spaces/brackets: "Machine failure", "Air temperature [K]". Use standard SQL. For "what tables" use SHOW TABLES IN {prefix}. For correlation matrix: use chart_type "correlation" and SQL that SELECTs all numeric columns (e.g. SELECT "Air temperature [K]", "Process temperature [K]", "Torque [Nm]" FROM table LIMIT 5000).

Example for "failure rates by machine type":
```json
{{"sql": "SELECT Type, SUM(\\"Machine failure\") as failures, COUNT(*) as total FROM {prefix}.cnc_data_ai_4_i_2020 GROUP BY Type", "chart_type": "bar", "explanation": "Failure counts by CNC machine type (H, L, M)"}}
```

If the question is NOT about data, respond in plain text without JSON."""


def _build_chat_prompt(dynamic_schema: str | None = None) -> str:
    data_context = get_schema_context(dynamic_context=dynamic_schema)
    return f"""You are a data assistant for DataKnobs predictive maintenance data. Help users explore CNC machines, electrical faults, batteries, turbofan engines, and power transformers.

{data_context}

When users ask vaguely (e.g. "visualize cnc", "show me data", "exploratory analysis"), suggest specific prompts they can try:
- "Show me CNC machine failure rates by type (H, L, M)"
- "Plot failure type distribution (TWF, HDF, PWF, OSF, RNF)"
- "Battery capacity degradation over cycles"
- "Electrical fault counts by phase"
- "Transformer oil temperature vs winding temperature over time"

Answer questions, suggest analyses, and help interpret results. Be concise."""


def run_data_query(
    user_message: str,
    chat_history: list,
    dynamic_schema: str | None = None,
) -> tuple[str, object | None, "pd.DataFrame | None", str]:
    """
    Ask the LLM to generate SQL + chart, execute it, and return response + Plotly figure.
    """
    system_prompt = _build_data_prompt(dynamic_schema=dynamic_schema)
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

    sql_query = clean_sql(parsed.get("sql", "") or "")
    chart_type = parsed.get("chart_type", "bar")
    explanation = parsed.get("explanation", "")

    if not sql_query or len(sql_query.strip()) < 10:
        return (
            f"**No valid SQL generated.** The model returned empty or invalid SQL.\n\n"
            f"**Raw response:** {response[:500]}...\n\n"
            "Try rephrasing your question or use a prompt like: *Show me CNC machine failure rates by type*",
            None, None, "bar"
        )

    try:
        df = run_sql_query(sql_query)
        st.session_state.last_sql = sql_query
        st.session_state.query_history = [sql_query] + [q for q in st.session_state.query_history if q != sql_query][:9]
    except Exception as e:
        logger.exception("SQL execution failed")
        err = str(e)
        hint = (
            "**Troubleshooting:**\n"
            "1. **SQL warehouse** – Ensure it's started (SQL Warehouses → select → Start)\n"
            "2. **App permissions** – App service principal needs **Can Use** on the warehouse\n"
            "3. **Unity Catalog** – Tables must exist; app needs `USE CATALOG`, `USE SCHEMA`, `SELECT` on tables\n"
            "4. **Connection** – 'Error during request to server' often means warehouse unreachable or auth failed"
        )
        return f"**SQL execution failed:** {err}\n\n**Query attempted:**\n```sql\n{sql_query}\n```\n\n{hint}", None, None, "bar"

    try:
        if (chart_type or "").lower() == "correlation":
            from visualization import create_correlation_heatmap
            fig = create_correlation_heatmap(df, "Correlation matrix")
        else:
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


def run_chat(user_message: str, chat_history: list, dynamic_schema: str | None = None) -> str:
    """General chat without data visualization."""
    system_prompt = _build_chat_prompt(dynamic_schema=dynamic_schema)
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

    warehouse_id = os.getenv("DATABRICKS_WAREHOUSE_ID")

    # Sidebar: example prompts + schema refresh
    with st.sidebar:
        st.subheader("📊 DataKnobs datasets")
        st.caption("CNC, electrical faults, battery, turbofan, transformer")
        if warehouse_id:
            if st.button("🔄 Load schema from database", help="Fetch actual column names and sample rows to improve SQL generation"):
                with st.spinner("Fetching schema..."):
                    try:
                        st.session_state.cached_schema = get_dynamic_schema_context()
                        if st.session_state.cached_schema:
                            st.success("Schema loaded")
                        else:
                            st.warning("Could not fetch schema. Tables may not exist yet.")
                    except Exception as e:
                        st.error(f"Failed: {e}")
                st.rerun()
            if st.session_state.get("cached_schema"):
                st.caption("✓ Using live schema")
        st.divider()
        st.caption("Try these prompts:")
        for ex in EXAMPLE_PROMPTS[:5]:
            if st.button(ex, key=ex[:20], use_container_width=True):
                st.session_state.suggested_prompt = ex
                st.rerun()

        st.divider()
        st.subheader("📜 Query history")
        for i, q in enumerate((st.session_state.query_history or [])[:5]):
            q_short = (q[:60] + "…") if len(q) > 60 else q
            if st.button(q_short.replace("\n", " "), key=f"qh_{i}", use_container_width=True):
                st.session_state.raw_sql_to_run = q
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

    if not warehouse_id:
        st.warning(
            "**DATABRICKS_WAREHOUSE_ID** is not set. Data visualization will not work. "
            "Add a SQL warehouse resource in app.yaml."
        )

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "viz_data" not in st.session_state:
        st.session_state.viz_data = None
    if "viz_chart_type" not in st.session_state:
        st.session_state.viz_chart_type = "bar"
    if "cached_schema" not in st.session_state:
        st.session_state.cached_schema = None
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    if "last_sql" not in st.session_state:
        st.session_state.last_sql = None
    if "show_correlation" not in st.session_state:
        st.session_state.show_correlation = False

    dynamic_schema = st.session_state.get("cached_schema")

    # Use suggested prompt from sidebar button or raw SQL from history
    prompt_input = st.chat_input("Ask about failure rates, asset health, or trends...")
    if "suggested_prompt" in st.session_state:
        prompt_input = st.session_state.pop("suggested_prompt", None)

    # Raw SQL mode - check if we have SQL to run from history or raw editor
    raw_sql = st.session_state.pop("raw_sql_to_run", None)

    # Two-column layout: chat left, interactive viz right
    chat_col, viz_col = st.columns([3, 2], gap="large")

    with chat_col:
        st.subheader("💬 Chat")

        # Raw SQL mode
        with st.expander("🔧 Run raw SQL", expanded=False):
            sql_editor = st.text_area(
                "Enter SQL",
                value=st.session_state.get("last_sql", "SELECT * FROM dataknobs_predictive_maintenance_and_asset_management.datasets.cnc_data_ai_4_i_2020 LIMIT 100"),
                height=120,
                key="raw_sql_editor",
            )
            if st.button("Run SQL", key="run_raw_sql") or raw_sql:
                sql_to_run = raw_sql or sql_editor
                if warehouse_id and sql_to_run.strip():
                    try:
                        df_raw = run_sql_query(clean_sql(sql_to_run))
                        st.session_state.viz_data = df_raw
                        st.session_state.viz_chart_type = "bar"
                        st.session_state.last_sql = sql_to_run
                        st.session_state.query_history = [sql_to_run] + [q for q in st.session_state.query_history if q != sql_to_run][:9]
                        st.success(f"Query returned {len(df_raw)} rows")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

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
                        response_text, fig, df, chart_type = run_data_query(
                            prompt, st.session_state.messages, dynamic_schema=dynamic_schema
                        )
                        if df is not None and fig is not None:
                            st.session_state.viz_data = df
                            st.session_state.viz_chart_type = chart_type
                    else:
                        response_text = run_chat(
                            prompt, st.session_state.messages, dynamic_schema=dynamic_schema
                        )
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

            chart_opts = ["bar", "line", "scatter", "pie", "heatmap", "correlation"]
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
                if chart_type == "correlation":
                    viz_fig = create_correlation_heatmap(df, f"Correlation matrix ({len(df)} rows)")
                else:
                    viz_fig = create_chart_with_selection(
                        df, chart_type,
                        x_col=x_val, y_col=y_val, color_col=color_val,
                        title=f"Data: {len(df)} rows",
                    )
                st.plotly_chart(viz_fig, use_container_width=True, key="viz_chart")
            except Exception as e:
                st.error(f"Chart error: {e}")
                viz_fig = infer_chart_from_data(df, "bar" if chart_type == "correlation" else chart_type)
                st.plotly_chart(viz_fig, use_container_width=True, key="viz_chart_fallback")

            with st.expander("📋 Data table", expanded=False):
                st.dataframe(df.head(100), use_container_width=True, height=300)

            # Action buttons
            btn_col1, btn_col2, btn_col3, btn_col4, btn_col5 = st.columns(5)
            with btn_col1:
                csv = df.to_csv(index=False)
                st.download_button("📥 Export CSV", csv, file_name="data.csv", mime="text/csv", key="dl_csv")
            with btn_col2:
                try:
                    png_bytes = viz_fig.to_image(format="png")
                    st.download_button("📥 Download PNG", png_bytes, file_name="chart.png", mime="image/png", key="dl_png")
                except Exception:
                    st.caption("PNG (kaleido)")
            with btn_col3:
                numeric_count = len([c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])])
                if numeric_count >= 2 and st.button("📊 Correlation", key="corr_btn"):
                    st.session_state.show_correlation = True
                    st.rerun()
            with btn_col4:
                if st.session_state.get("last_sql") and st.button("🔄 Refresh", key="refresh_btn", help="Re-run last query"):
                    st.session_state.raw_sql_to_run = st.session_state.last_sql
                    st.rerun()
            with btn_col5:
                if st.button("Clear", key="viz_clear"):
                    st.session_state.viz_data = None
                    st.session_state.show_correlation = False
                    st.rerun()

            if st.session_state.get("show_correlation"):
                st.divider()
                corr_fig = create_correlation_heatmap(df, "Correlation matrix")
                st.plotly_chart(corr_fig, use_container_width=True, key="corr_chart")
        else:
            st.info("Ask a data question to see visualizations here. Try: *Show me CNC machine failure rates by type*")
            st.markdown("**Example prompts:**")
            for ex in EXAMPLE_PROMPTS[:3]:
                st.markdown(f"- {ex}")


if __name__ == "__main__":
    main()
