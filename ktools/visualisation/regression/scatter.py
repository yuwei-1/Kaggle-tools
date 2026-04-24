import plotly.graph_objects as go
import numpy as np
import pandas as pd


def plot_professional_scatter(
    x: pd.Series,
    y: pd.Series,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    show_trendline: bool = True,
    marker_size: int = 8,
    marker_opacity: float = 0.7,
) -> go.Figure:
    """
    Create a professional-looking scatter plot between two numerical variables.

    Parameters:
    -----------
    x : pd.Series
        Series containing values for the x-axis
    y : pd.Series
        Series containing values for the y-axis
    title : str, optional
        Plot title (default: inferred from series names)
    xlabel : str, optional
        X-axis label (default: x.name or 'X')
    ylabel : str, optional
        Y-axis label (default: y.name or 'Y')
    show_trendline : bool, optional
        Whether to show a linear trendline (default: True)
    marker_size : int, optional
        Size of scatter markers (default: 8)
    marker_opacity : float, optional
        Opacity of scatter markers (default: 0.7)

    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The plotly figure object
    """
    # Infer labels from series names if not provided
    if xlabel is None:
        xlabel = x.name if x.name else "X"
    if ylabel is None:
        ylabel = y.name if y.name else "Y"
    if title is None:
        if x.name and y.name:
            title = f"{y.name} vs {x.name}"
        else:
            title = "Scatter Plot"

    x_arr = np.array(x)
    y_arr = np.array(y)

    fig = go.Figure()

    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
    primary_color = colors[0]

    # Add scatter plot
    fig.add_trace(
        go.Scatter(
            x=x_arr,
            y=y_arr,
            mode="markers",
            name="Data Points",
            marker=dict(
                size=marker_size,
                color=primary_color,
                opacity=marker_opacity,
                line=dict(width=1, color="white"),
            ),
            hovertemplate=f"{xlabel}: %{{x}}<br>{ylabel}: %{{y}}<extra></extra>",
        )
    )

    # Add trendline if requested
    if show_trendline:
        # Calculate linear regression
        mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
        x_clean = x_arr[mask]
        y_clean = y_arr[mask]

        if len(x_clean) > 1:
            coeffs = np.polyfit(x_clean, y_clean, 1)
            slope, intercept = coeffs
            x_line = np.array([x_clean.min(), x_clean.max()])
            y_line = slope * x_line + intercept

            # Calculate R-squared
            y_pred = slope * x_clean + intercept
            ss_res = np.sum((y_clean - y_pred) ** 2)
            ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name=f"Trendline (R² = {r_squared:.3f})",
                    line=dict(color=colors[1], width=2.5, dash="dash"),
                    hoverinfo="skip",
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
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#bdc3c7",
            borderwidth=1,
            font=dict(size=12, color="#34495e"),
        ),
        width=900,
        height=600,
        margin=dict(l=80, r=50, t=100, b=80),
        font=dict(family="Arial, sans-serif", color="#2c3e50"),
        hovermode="closest",
    )

    return fig
