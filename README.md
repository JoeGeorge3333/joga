# Predictive Maintenance Chatbot – Databricks App

A native **Databricks App** with a chatbot that queries and visualizes predictive maintenance data across 5 industrial asset types. Built for the [Predictive Maintenance & Asset Management Data Platform](https://www.dataknobs.com/).

## Features

- **Chat interface** – Ask questions in natural language about asset health, failure rates, and trends
- **Data visualization** – The chatbot generates SQL, runs it, and renders Plotly charts (bar, line, scatter, pie, heatmap)
- **5 asset domains** – CNC machines, electrical faults, battery/heater degradation, NASA turbofan RUL, power transformer health
- **Native Databricks** – Runs on Databricks Apps with SQL warehouse and model serving integration

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Databricks App (Streamlit)                 │
├─────────────────────────────────────────────────────────────┤
│  Chat UI  │  LLM (Foundation Model / Agent)  │  SQL + Charts  │
│           │  - Generates SQL from user query │  - Databricks │
│           │  - Returns chart_type + explanation               │
│           │                                    │  - Plotly    │
└─────────────────────────────────────────────────────────────┘
         │                              │                    │
         ▼                              ▼                    ▼
   Model Serving              Unity Catalog            SQL Warehouse
   Endpoint                   (Predictive Maintenance  (Query execution)
                              datasets)
```

## Prerequisites

1. **Databricks workspace** with:
   - Model Serving (foundation model or agent endpoint)
   - SQL warehouse
   - Unity Catalog with predictive maintenance data

2. **Data** – Load the predictive maintenance datasets into Unity Catalog. The app expects tables under a configurable catalog/schema (default: `dataknobs_predictive_maintenance_and_asset_management.datasets`).

   Tables:
   - `cnc_data_ai_4_i_2020` – CNC machine failure
   - `electrical_fault_train_test_data` / `electrical_fault_validation_data`
   - `heater_train_test_data` / `heater_validation_data`
   - `nasa_data_train_test` / `nasa_data_validation`
   - `transformer_train_test_data` / `transformer_validation_data`

## Quick Start

### 1. Create the app in Databricks

1. In your workspace, go to **Apps** → **+ New App**
2. Choose **Import from Git** or **Upload** and point to this repo
3. Or create from template and replace the source with this code

### 2. Add resources

In the app configuration, add:

| Resource           | Key (default)     | Permission | Env var used in app      |
|--------------------|-------------------|------------|--------------------------|
| Model serving      | `serving-endpoint`| Can query  | `SERVING_ENDPOINT`       |
| SQL warehouse      | `sql-warehouse`   | Can use    | `DATABRICKS_WAREHOUSE_ID`|

### 3. Configure data location (optional)

If your data is in a different catalog/schema, set:

- `DATA_CATALOG` – e.g. `my_catalog`
- `DATA_SCHEMA` – e.g. `predictive_maintenance`

### 4. Deploy and run

Deploy the app from the Databricks UI or CLI. The app will start and provide a URL.

## Local development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables (replace with your actual values)
export SERVING_ENDPOINT="databricks-meta-llama-3-3-70b-instruct"  # or your endpoint name from Model Serving
export DATABRICKS_WAREHOUSE_ID="abc123def456"  # your SQL warehouse ID
export DATA_CATALOG="dataknobs_predictive_maintenance_and_asset_management"
export DATA_SCHEMA="datasets"

# Authenticate (Databricks CLI)
databricks auth login

# Run locally
streamlit run app.py
```

Or use Databricks CLI:

```bash
databricks apps run-local --prepare-environment --debug
```

## Example prompts

- *"Show me CNC machine failure rates by type (H, L, M)"*
- *"What's the distribution of failure types (TWF, HDF, PWF, OSF, RNF)?"*
- *"Plot battery capacity degradation over cycles"*
- *"Compare oil temperature vs winding temperature for the transformer"*
- *"How many electrical faults are there by phase (A, B, C, Ground)?"*

## Project structure

```
joga/
├── app.py              # Main Streamlit app
├── app.yaml            # Databricks App runtime config
├── requirements.txt    # Python dependencies
├── utils.py            # SQL execution, schema context
├── model_serving_utils.py  # Chat endpoint client
├── visualization.py    # Plotly chart generation
└── README.md
```

## Supported endpoints

The chatbot works with:

- **Foundation models** – e.g. `databricks-meta-llama-3-3-70b-instruct`, `databricks-dbrx-instruct`
- **Agent endpoints** – Custom agents deployed via `agents.deploy()` with chat task type

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "SERVING_ENDPOINT not set" | Add a model serving resource in app config |
| "DATABRICKS_WAREHOUSE_ID not set" | Add a SQL warehouse resource |
| "Table not found" | Verify data is in Unity Catalog; check `DATA_CATALOG` and `DATA_SCHEMA` |
| Charts not rendering | Ensure the LLM returns valid JSON with `sql`, `chart_type`, `explanation` |

## References

- [Databricks Apps](https://docs.databricks.com/en/dev-tools/databricks-apps/)
- [Chat UI with Databricks Apps](https://docs.databricks.com/en/generative-ai/agent-framework/chat-app)
- [Predictive Maintenance Documentation](.) (see PDF)
