import plotly.graph_objects as go
import numpy as np
import pandas as pd


def plot_professional_boxplot(
    categories: pd.Series, values: pd.Series, title: str | None = None
) -> go.Figure:
    """
    Create a professional-looking box plot from categorical and numerical data.

    Parameters:
    -----------
    categories : array-like
        Array containing categorical variable (aligned with values)
    values : array-like
        Array containing numerical values (aligned with categories)
    title : str, optional
        Plot title (default: 'Distribution Analysis by Category')

    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The plotly figure object
    """

    xlabel = categories.name
    ylabel = values.name
    if title is None:
        title = f"{ylabel} Distribution by {xlabel}"

    categories = np.array(categories)
    values = np.array(values)

    fig = go.Figure()
    unique_cats = np.unique(categories)

    for cat in unique_cats:
        cat_values = values[categories == cat]
        fig.add_trace(
            go.Box(
                y=cat_values,
                name=str(cat),
                boxmean=False,
                marker=dict(
                    size=3, opacity=0.5, line=dict(width=0.5, color="rgba(0,0,0,0.3)")
                ),
                line=dict(width=2.5),
                fillcolor="rgba(255,255,255,0.8)",
                notched=False,
                whiskerwidth=0.7,
            )
        )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=24, family="Arial, sans-serif", color="#2c3e50"),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            title=xlabel,
            title_font=dict(size=16, family="Arial, sans-serif", color="#34495e"),
            tickfont=dict(size=13, color="#34495e"),
            showgrid=False,
            showline=True,
            linewidth=2,
            linecolor="#bdc3c7",
            mirror=True,
        ),
        yaxis=dict(
            title=ylabel,
            title_font=dict(size=16, family="Arial, sans-serif", color="#34495e"),
            tickfont=dict(size=13, color="#34495e"),
            showgrid=True,
            gridwidth=1,
            gridcolor="#ecf0f1",
            showline=True,
            linewidth=2,
            linecolor="#bdc3c7",
            mirror=True,
            zeroline=False,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        width=900,
        height=600,
        margin=dict(l=80, r=50, t=100, b=80),
        font=dict(family="Arial, sans-serif", color="#2c3e50"),
        boxmode="group",
        hovermode="closest",
    )

    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
    for i, trace in enumerate(fig.data):
        trace.marker.color = colors[i % len(colors)]
        trace.line.color = colors[i % len(colors)]

    return fig
