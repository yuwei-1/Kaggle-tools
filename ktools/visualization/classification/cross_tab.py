import plotly.graph_objects as go
import pandas as pd


def plot_cross_tabulation(categories: pd.Series, values: pd.Series, 
                            title: str = None, xlabel: str = None, ylabel: str = None) -> go.Figure:
    """
    Create a professional-looking grouped bar plot showing counts of categorical values
    for each target feature value.
    
    Parameters:
    -----------
    categories : pd.Series
        Series containing categorical variable to analyze
    target : pd.Series
        Series containing target feature (creates separate bars for each unique value)
    title : str, optional
        Plot title (default: inferred from series names)
    xlabel : str, optional
        X-axis label (default: categories.name or 'Category')
    ylabel : str, optional
        Y-axis label (default: 'Count')
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The plotly figure object
    """
    # Infer labels from series names if not provided
    if xlabel is None:
        xlabel = categories.name if categories.name else 'Category'
    if ylabel is None:
        ylabel = 'Count'
    if title is None:
        if categories.name and values.name:
            title = f'{categories.name} Distribution by {values.name}'
        else:
            title = 'Categorical Distribution by Target'
    
    # Create a crosstab to get counts
    counts_df = pd.crosstab(categories, values)
    normaliser = counts_df.sum(axis=1)
    
    fig = go.Figure()
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    # Add a bar trace for each target value

    
    for i, target_val in enumerate(counts_df.columns):
        fig.add_trace(go.Bar(
            x=counts_df.index,
            y=counts_df[target_val] / normaliser,
            name=str(target_val),
            marker=dict(
                color=colors[i % len(colors)],
                line=dict(color='white', width=1)
            ),
            opacity=0.85
        ))
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=24, family='Arial, sans-serif', color='#2c3e50'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=xlabel,
            title_font=dict(size=16, family='Arial, sans-serif', color='#34495e'),
            tickfont=dict(size=13, color='#34495e'),
            showgrid=False,
            showline=True,
            linewidth=2,
            linecolor='#bdc3c7',
            mirror=True
        ),
        yaxis=dict(
            title=ylabel,
            title_font=dict(size=16, family='Arial, sans-serif', color='#34495e'),
            tickfont=dict(size=13, color='#34495e'),
            showgrid=True,
            gridwidth=1,
            gridcolor='#ecf0f1',
            showline=True,
            linewidth=2,
            linecolor='#bdc3c7',
            mirror=True,
            zeroline=False
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=900,
        height=600,
        margin=dict(l=80, r=50, t=100, b=80),
        font=dict(family='Arial, sans-serif', color='#2c3e50'),
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1,
        legend=dict(
            title=dict(
                text=values.name if values.name else 'Target',
                font=dict(size=14, family='Arial, sans-serif', color='#2c3e50')
            ),
            font=dict(size=12, color='#34495e'),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#bdc3c7',
            borderwidth=1
        ),
        hovermode='x unified'
    )
    
    return fig