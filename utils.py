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


def get_schema_context() -> str:
    """Return schema documentation for the LLM to generate SQL."""
    prefix = _get_table_prefix()
    return f"""
## Predictive Maintenance Datasets (Unity Catalog)

Table prefix: {prefix}

### 1. CNC Machine Failure (cnc_data_ai_4_i_2020)
- UDI (bigint), Product ID (string), Type (string: H, L, M)
- Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min]
- Machine failure (bigint: 0/1), TWF, HDF, PWF, OSF, RNF (failure type flags)

### 2. Electrical Fault (electrical_fault_train_test_data, electrical_fault_validation_data)
- G, C, B, A (fault flags), Ia, Ib, Ic (currents), Va, Vb, Vc (voltages)

### 3. Battery/Heater (heater_train_test_data, heater_validation_data)
- Voltage_measured, Current_measured, Temperature_measured, Current_charge, Voltage_charge
- Time, Capacity, id_cycle, type, ambient_temperature, time_year, PhID

### 4. NASA Turbofan (nasa_data_train_test, nasa_data_validation)
- id, Cycle, OpSet1, OpSet2, OpSet3, SensorMeasure1-21, RemainingUsefulLife

### 5. Power Transformer (transformer_train_test_data, transformer_validation_data)
- DeviceTimeStamp, OTI, WTI, ATI, OLI, OTI_A, OTI_T
- VL1, VL2, VL3, IL1, IL2, IL3, VL12, VL23, VL31, INUT, MOG_A
"""


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
