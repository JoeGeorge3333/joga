"""Utility functions for the Predictive Maintenance Chatbot app."""

import json
import logging
import os
import re
from typing import Any

import pandas as pd
from databricks import sql
from databricks.sdk.core import Config

logger = logging.getLogger(__name__)

# Schema prefix for Unity Catalog tables
def _get_table_prefix() -> str:
    catalog = os.getenv("DATA_CATALOG", "dataknobs_predictive_maintenance_and_asset_management")
    schema = os.getenv("DATA_SCHEMA", "datasets")
    return f"{catalog}.{schema}"


def run_sql_query(query: str) -> pd.DataFrame:
    """Execute a SQL query against the Databricks SQL warehouse."""
    warehouse_id = os.getenv("DATABRICKS_WAREHOUSE_ID")
    if not warehouse_id:
        raise ValueError("DATABRICKS_WAREHOUSE_ID must be set in app.yaml")

    cfg = Config()
    with sql.connect(
        server_hostname=cfg.host,
        http_path=f"/sql/1.0/warehouses/{warehouse_id}",
        credentials_provider=cfg.authenticate,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall_arrow().to_pandas()


# Fallback table names if SHOW TABLES fails
DATAKNOBS_TABLES = [
    "cnc_data_ai_4_i_2020",
    "electrical_fault_train_test_data",
    "electrical_fault_validation_data",
    "heater_train_test_data",
    "heater_validation_data",
    "nasa_data_train_test",
    "nasa_data_validation",
    "transformer_train_test_data",
    "transformer_validation_data",
]


def fetch_table_names() -> list[str]:
    """Fetch actual table names from the database. Returns empty list on failure."""
    prefix = _get_table_prefix()
    try:
        df = run_sql_query(f"SHOW TABLES IN {prefix}")
        if df.empty:
            return []
        # SHOW TABLES returns database, tableName, isTemporary
        name_col = "tableName" if "tableName" in df.columns else df.columns[1]
        return [str(row[name_col]).strip() for _, row in df.iterrows() if row.get(name_col)]
    except Exception as e:
        logger.warning("Could not fetch table names: %s", e)
        return []


def fetch_table_schema(table_name: str) -> str | None:
    """Fetch actual column names and types from the database. Returns None on failure."""
    prefix = _get_table_prefix()
    full_name = f"{prefix}.{table_name}"
    try:
        df = run_sql_query(f"DESCRIBE TABLE {full_name}")
        if df.empty:
            return None
        # DESCRIBE returns col_name, data_type, comment (columns may vary by dialect)
        cols = list(df.columns)
        name_col = cols[0] if cols else "col_name"
        type_col = cols[1] if len(cols) > 1 else "data_type"
        lines = [f"- {row[name_col]} ({row[type_col]})" for _, row in df.iterrows()]
        return "\n".join(lines)
    except Exception as e:
        logger.warning("Could not fetch schema for %s: %s", table_name, e)
        return None


def fetch_sample_rows(table_name: str, limit: int = 3) -> str | None:
    """Fetch sample rows to show data format. Returns None on failure."""
    prefix = _get_table_prefix()
    full_name = f"{prefix}.{table_name}"
    try:
        df = run_sql_query(f"SELECT * FROM {full_name} LIMIT {limit}")
        if df.empty:
            return None
        return df.to_string(max_colwidth=20)
    except Exception as e:
        logger.warning("Could not fetch samples for %s: %s", table_name, e)
        return None


def get_dynamic_schema_context() -> str | None:
    """
    Fetch actual schema + sample rows from the database.
    Returns enriched context string, or None if fetch fails.
    """
    prefix = _get_table_prefix()
    tables = fetch_table_names() or DATAKNOBS_TABLES[:5]
    tables = tables[:7]  # Limit to avoid timeout
    parts = [f"## Live schema from database (prefix: {prefix})\n"]
    success = False
    for table in tables:
        schema = fetch_table_schema(table)
        if schema:
            success = True
            parts.append(f"### {table}\n{schema}\n")
            samples = fetch_sample_rows(table)
            if samples:
                parts.append(f"Sample rows:\n```\n{samples}\n```\n")
    return "\n".join(parts) if success else None


CATALOG_DESCRIPTION = """
The catalog organizes data related to predictive maintenance and asset management. It includes schemas for equipment performance metrics, maintenance schedules, failure predictions, and asset lifecycle tracking. Use this catalog for analyzing maintenance needs, optimizing asset utilization, and supporting operational efficiency.
"""


def get_schema_context(dynamic_context: str | None = None) -> str:
    """Return schema documentation for the LLM. Pass dynamic_context from fetch when available."""
    prefix = _get_table_prefix()
    static = f"""
## DataKnobs Predictive Maintenance Datasets (Unity Catalog)

**Catalog purpose:** {CATALOG_DESCRIPTION.strip()}

Table prefix: {prefix}

### 1. CNC Machine Failure ({prefix}.cnc_data_ai_4_i_2020)
- UDI, Product ID, Type (H, L, M), "Air temperature [K]", "Process temperature [K]"
- "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "Machine failure", TWF, HDF, PWF, OSF, RNF

### 2. Electrical Fault ({prefix}.electrical_fault_train_test_data, {prefix}.electrical_fault_validation_data)
- G, C, B, A (fault flags), Ia, Ib, Ic (currents), Va, Vb, Vc (voltages)

### 3. Battery/Heater ({prefix}.heater_train_test_data, {prefix}.heater_validation_data)
- Voltage_measured, Current_measured, Temperature_measured, Capacity, id_cycle, PhID, Time

### 4. NASA Turbofan ({prefix}.nasa_data_train_test, {prefix}.nasa_data_validation)
- id, Cycle, OpSet1, OpSet2, OpSet3, SensorMeasure1-21, RemainingUsefulLife

### 5. Power Transformer ({prefix}.transformer_train_test_data, {prefix}.transformer_validation_data)
- DeviceTimeStamp, OTI, WTI, ATI, OLI, OTI_A, OTI_T, VL1, VL2, VL3, IL1, IL2, IL3, VL12, VL23, VL31, INUT, MOG_A

**SQL rules:** Column names with spaces or brackets MUST be double-quoted: "Machine failure", "Air temperature [K]"
"""
    if dynamic_context:
        return dynamic_context + "\n" + static
    return static


def clean_sql(sql: str) -> str:
    """Remove markdown code blocks and extra whitespace from SQL."""
    sql = sql.strip()
    if sql.startswith("```"):
        sql = re.sub(r"^```\w*\n?", "", sql)
        sql = re.sub(r"\n?```\s*$", "", sql)
    return sql.strip()


def parse_llm_data_response(response: str) -> dict[str, Any] | None:
    """
    Parse LLM response for SQL + chart request.
    Expects JSON block: {"sql": "...", "chart_type": "bar|line|scatter|pie", "explanation": "..."}
    """
    # Try to extract JSON from markdown code block or raw
    json_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", response)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try raw JSON in response
    json_match = re.search(r"\{[\s\S]*\"sql\"[\s\S]*\}", response)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return None
