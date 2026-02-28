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


def get_schema_context(dataset_mode: str = "all") -> str:
    """Return schema documentation for the LLM to generate SQL."""
    parts = []

    # Databricks sample datasets - available in every workspace
    if dataset_mode in ("all", "samples"):
        parts.append("""
## Databricks Sample Datasets (ALWAYS AVAILABLE - use these for real data)

### NYC Taxi (samples.nyctaxi.trips)
- tpep_pickup_datetime, tpep_dropoff_datetime, passenger_count, trip_distance
- fare_amount, total_amount, tip_amount, tolls_amount, PULocationID, DOLocationID
- payment_type, VendorID, RatecodeID
Example: SELECT fare_amount, trip_distance, passenger_count FROM samples.nyctaxi.trips LIMIT 1000

### TPC-H (samples.tpch)
- lineitem: l_orderkey, l_partkey, l_quantity, l_extendedprice, l_discount, l_shipdate
- orders: o_orderkey, o_custkey, o_totalprice, o_orderdate, o_orderstatus
- customer: c_custkey, c_name, c_nationkey
- part: p_partkey, p_name, p_retailprice
Example: SELECT o_orderdate, SUM(l_extendedprice) as revenue FROM samples.tpch.lineitem l JOIN samples.tpch.orders o ON l.l_orderkey = o.o_orderkey GROUP BY o_orderdate
""")

    # Predictive maintenance - when user has loaded this data
    if dataset_mode in ("all", "predictive"):
        prefix = _get_table_prefix()
        parts.append(f"""
## Predictive Maintenance Datasets (optional - prefix: {prefix})

### 1. CNC Machine Failure ({prefix}.cnc_data_ai_4_i_2020)
- UDI, Product ID, Type (H, L, M), Air temperature [K], Process temperature [K]
- Rotational speed [rpm], Torque [Nm], Tool wear [min], Machine failure, TWF, HDF, PWF, OSF, RNF

### 2. Electrical Fault ({prefix}.electrical_fault_train_test_data)
- G, C, B, A (fault flags), Ia, Ib, Ic (currents), Va, Vb, Vc (voltages)

### 3. Battery/Heater ({prefix}.heater_train_test_data)
- Voltage_measured, Current_measured, Temperature_measured, Capacity, id_cycle, PhID

### 4. NASA Turbofan ({prefix}.nasa_data_train_test)
- id, Cycle, OpSet1-3, SensorMeasure1-21, RemainingUsefulLife

### 5. Power Transformer ({prefix}.transformer_train_test_data)
- DeviceTimeStamp, OTI, WTI, ATI, OLI, VL1-3, IL1-3
""")

    return "\n".join(parts) if parts else "No datasets configured."


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
