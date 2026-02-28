"""Chart generation for predictive maintenance data."""

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_chart(
    df: pd.DataFrame,
    chart_type: str,
    x: str | None = None,
    y: str | None = None,
    color: str | None = None,
    title: str = "",
) -> go.Figure:
    """
    Create a Plotly chart from a DataFrame.
    chart_type: bar, line, scatter, pie, heatmap
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data to display", showarrow=False)
        return fig

    # Auto-select columns if not provided
    cols = list(df.columns)
    if not x and len(cols) >= 1:
        x = cols[0]
    if not y and len(cols) >= 2:
        y = cols[1]

    chart_type = (chart_type or "bar").lower()

    try:
        if chart_type == "bar":
            fig = px.bar(
                df,
                x=x or cols[0],
                y=y if y in df.columns else None,
                color=color if color in df.columns else None,
                title=title,
            )
        elif chart_type == "line":
            fig = px.line(
                df,
                x=x or cols[0],
                y=y if y and y in df.columns else [c for c in cols[1:] if pd.api.types.is_numeric_dtype(df[c])],
                color=color if color in df.columns else None,
                title=title,
            )
        elif chart_type == "scatter":
            fig = px.scatter(
                df,
                x=x or cols[0],
                y=y or (cols[1] if len(cols) > 1 else cols[0]),
                color=color if color in df.columns else None,
                title=title,
            )
        elif chart_type == "pie":
            fig = px.pie(
                df,
                names=x or cols[0],
                values=y if y and y in df.columns else None,
                title=title,
            )
        elif chart_type == "heatmap":
            numeric = df.select_dtypes(include=["number"])
            if numeric.empty:
                fig = px.bar(df, x=cols[0], title=title)
            else:
                fig = px.imshow(numeric.T, title=title, aspect="auto")
        else:
            fig = px.bar(df, x=x or cols[0], y=y, title=title)

        fig.update_layout(
            margin=dict(l=40, r=40, t=50, b=40),
            height=400,
            template="plotly_white",
        )
        return fig

    except Exception:
        # Fallback: simple table as bar chart
        fig = px.bar(df.head(20), x=cols[0], y=cols[1] if len(cols) > 1 else None, title=title)
        fig.update_layout(height=400, template="plotly_white")
        return fig


def infer_chart_from_data(df: pd.DataFrame, chart_type: str) -> go.Figure:
    """
    Infer sensible x/y/color from DataFrame and chart_type.
    """
    cols = list(df.columns)
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in cols if c not in numeric]

    x = categorical[0] if categorical else (numeric[0] if numeric else cols[0])
    y = numeric[0] if numeric and (not categorical or numeric[0] != x) else (numeric[1] if len(numeric) > 1 else None)
    color = categorical[1] if len(categorical) > 1 else None

    return create_chart(df, chart_type, x=x, y=y, color=color)


def create_chart_with_selection(
    df: pd.DataFrame,
    chart_type: str,
    x_col: str | None = None,
    y_col: str | None = None,
    color_col: str | None = None,
    title: str = "",
) -> go.Figure:
    """Create chart with explicit column selection for interactive EDA."""
    return create_chart(df, chart_type, x=x_col, y=y_col, color=color_col, title=title)
