import plotly.graph_objects as go
import numpy as np
import pandas as pd


def plot_professional_violin(
    categories: pd.Series,
    values: pd.Series,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
):
    """
    Create a professional-looking horizontal violin plot with mean and IQR markers.

    Parameters:
    -----------
    categories : pd.Series
        Series containing categorical variable (aligned with values)
    values : pd.Series
        Series containing numerical values (aligned with categories)
    title : str, optional
        Plot title (default: inferred from series names)
    xlabel : str, optional
        X-axis label (default: values.name or 'Value')
    ylabel : str, optional
        Y-axis label (default: categories.name or 'Category')

    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The plotly figure object
    """
    # Infer labels from series names if not provided
    if xlabel is None:
        xlabel = values.name if values.name else "Value"
    if ylabel is None:
        ylabel = categories.name if categories.name else "Category"
    if title is None:
        if categories.name and values.name:
            title = f"{values.name} by {categories.name}"
        else:
            title = "Distribution Analysis by Category"

    categories = np.array(categories)
    values = np.array(values)

    fig = go.Figure()
    unique_cats = np.unique(categories)

    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

    for i, cat in enumerate(unique_cats):
        cat_values = values[categories == cat]
        color = colors[i % len(colors)]

        # Add violin plot
        fig.add_trace(
            go.Violin(
                x=cat_values,
                y=[i] * len(cat_values),
                name=f"Distribution ({cat})",
                orientation="h",
                line_color=color,
                fillcolor=color,
                opacity=0.6,
                meanline_visible=False,
                points=False,
                width=0.7,
                showlegend=True,
            )
        )

        # Calculate statistics
        mean_val = np.mean(cat_values)
        q1 = np.percentile(cat_values, 25)
        q3 = np.percentile(cat_values, 75)

        # Add mean line (vertical) - bounded to violin width
        fig.add_trace(
            go.Scatter(
                x=[mean_val, mean_val],
                y=[i - 100, i + 100],
                mode="lines",
                line=dict(color=color, width=2, dash="dash"),
                showlegend=True,
                name=f"Mean ({cat})",
                hoverinfo="skip",
            )
        )

        # Add IQR range (Q1 to Q3)
        fig.add_trace(
            go.Scatter(
                x=[q1, q3],
                y=[i, i],
                mode="lines",
                line=dict(color=color, width=2, dash="solid"),
                showlegend=True,
                name=f"IQR ({cat})",
                hoverinfo="skip",
            )
        )

        # Add Q1 marker
        fig.add_trace(
            go.Scatter(
                x=[q1],
                y=[i],
                mode="markers",
                marker=dict(
                    size=8, color=color, symbol="line-ns-open", line=dict(width=2)
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Add Q3 marker
        fig.add_trace(
            go.Scatter(
                x=[q3],
                y=[i],
                mode="markers",
                marker=dict(
                    size=8, color=color, symbol="line-ns-open", line=dict(width=2)
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Set y-axis range to prevent extension from vertical lines
    y_range = [-1, len(unique_cats)]

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
            showgrid=True,
            gridwidth=1,
            gridcolor="#ecf0f1",
            showline=True,
            linewidth=2,
            linecolor="#bdc3c7",
            mirror=True,
            zeroline=False,
        ),
        yaxis=dict(
            title=ylabel,
            title_font=dict(size=16, family="Arial, sans-serif", color="#34495e"),
            tickfont=dict(size=13, color="#34495e"),
            showgrid=False,
            showline=True,
            linewidth=2,
            linecolor="#bdc3c7",
            mirror=True,
            range=y_range,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=900,
        height=600,
        margin=dict(l=80, r=50, t=100, b=80),
        font=dict(family="Arial, sans-serif", color="#2c3e50"),
        hovermode="closest",
        violinmode="overlay",
    )

    return fig
