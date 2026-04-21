import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from ktools.experiment.validation_selector import ValidationSelector
from ktools.config.dataset import DatasetConfig
from ktools.models.gbdt.lightgbm import LGBMModel


DATA_PATH = "/workspaces/Kaggle-tools/data/diabetes_prediction/train.csv"
TARGET_COL = "diagnosed_diabetes"

data = pd.read_csv(DATA_PATH)

numerical_cols = data.select_dtypes(include=["number"]).columns.tolist()
numerical_cols = [col for col in numerical_cols if col not in ["id", TARGET_COL]]
categorical_cols = data.select_dtypes(include=["object"]).columns.tolist()
training_cols = numerical_cols + categorical_cols

for col in categorical_cols:
    data[col] = data[col].astype("category")

config = DatasetConfig(
    training_col_names=training_cols,
    target_col_name=TARGET_COL,
    numerical_col_names=numerical_cols,
    categorical_col_names=categorical_cols,
)

model = LGBMModel(
    num_boost_round=100,
    early_stopping_rounds=50,
    random_state=42,
)

selector_5fold = ValidationSelector(
    model=model,
    outer_fold_splitter=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    inner_fold_splitter=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    metric=roc_auc_score,
)

selector_10fold = ValidationSelector(
    model=model,
    outer_fold_splitter=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    inner_fold_splitter=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
    metric=roc_auc_score,
)

results_5fold = selector_5fold.run(data, config)
results_10fold = selector_10fold.run(data, config)

fig = go.Figure()

colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

for fold_idx, (fold_name, fold_data) in enumerate(results_5fold.items()):
    inner_scores = fold_data["inner_cv_scores"]
    test_score = fold_data["simulated_test_set_score"]

    fig.add_trace(
        go.Box(
            y=inner_scores,
            name=f"5-Fold Inner CV ({fold_name})",
            marker_color=colors[fold_idx % len(colors)],
            boxmean=True,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[f"5-Fold Inner CV ({fold_name})"],
            y=[test_score],
            mode="markers",
            name=f"5-Fold Test ({fold_name})",
            marker=dict(
                size=15,
                color=colors[fold_idx % len(colors)],
                symbol="diamond",
                line=dict(width=2, color="black"),
            ),
        )
    )

for fold_idx, (fold_name, fold_data) in enumerate(results_10fold.items()):
    inner_scores = fold_data["inner_cv_scores"]
    test_score = fold_data["simulated_test_set_score"]

    fig.add_trace(
        go.Box(
            y=inner_scores,
            name=f"10-Fold Inner CV ({fold_name})",
            marker_color=colors[fold_idx % len(colors)],
            boxmean=True,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[f"10-Fold Inner CV ({fold_name})"],
            y=[test_score],
            mode="markers",
            name=f"10-Fold Test ({fold_name})",
            marker=dict(
                size=15,
                color=colors[fold_idx % len(colors)],
                symbol="star",
                line=dict(width=2, color="black"),
            ),
        )
    )

fig.update_layout(
    title=dict(
        text="Comparison of 5-Fold vs 10-Fold Stratified KFold CV Scores",
        font=dict(size=24, family="Arial, sans-serif", color="#2c3e50"),
        x=0.5,
        xanchor="center",
    ),
    xaxis=dict(
        title="Fold Configuration",
        title_font=dict(size=16, family="Arial, sans-serif", color="#34495e"),
        tickfont=dict(size=11, color="#34495e"),
        showgrid=False,
        showline=True,
        linewidth=2,
        linecolor="#bdc3c7",
        mirror=True,
    ),
    yaxis=dict(
        title="ROC AUC Score",
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
    width=1400,
    height=700,
    margin=dict(l=80, r=50, t=100, b=150),
    font=dict(family="Arial, sans-serif", color="#2c3e50"),
)

fig.write_image("data/diabetes_prediction/validation_comparison.png")
fig.show()
