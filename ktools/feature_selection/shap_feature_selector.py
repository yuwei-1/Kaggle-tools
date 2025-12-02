from typing import Tuple, Union
import shap
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class SHAPFeatureSelector:

    def __init__(self,
                 trained_model : BaseEstimator,
                 top_k_features : int = 20,
                 feature_importance_save_path : Union[str, None] = None) -> None:
        
        self._trained_model = trained_model
        self._top_k_features = top_k_features
        self._feature_importance_save_path = feature_importance_save_path

    def select(self, X : pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:

        training_features = X.columns.tolist()
        explainer = shap.TreeExplainer(self._trained_model)
        shap_values = explainer.shap_values(X)
        aggregated_shap_values = np.abs(shap_values).mean(0)

        idcs = np.argsort(aggregated_shap_values)
        sorted_features = np.array(training_features)[idcs][::-1]
        sorted_shap_values = aggregated_shap_values[idcs][::-1]

        self._plot_shap_values(sorted_features[:self._top_k_features], sorted_shap_values[:self._top_k_features])
        return sorted_features, sorted_shap_values

    
    def _plot_shap_values(self, top_features : np.ndarray, top_importances: np.ndarray) -> None:

        norm = mcolors.Normalize(vmin=top_importances.min(), vmax=top_importances.max())
        cmap = plt.get_cmap('viridis')
        colors = cmap(norm(top_importances))

        plt.figure(figsize=(10, 8))
        plt.barh(top_features, top_importances, color=colors, edgecolor='black', height=0.5)
        plt.xlabel("Mean |SHAP value|", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.title("Top 20 Feature Importances by Mean Absolute SHAP Value", fontsize=14)
        plt.gca().invert_yaxis()
        plt.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        if self._feature_importance_save_path:
            plt.savefig(self._feature_importance_save_path, dpi=300)
        plt.show()