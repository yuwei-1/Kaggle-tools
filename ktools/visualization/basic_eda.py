import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder


class BasicDatasetEDA:
    def __init__(self, train_df, test_df, target_col_name) -> None:
        self._train_df = train_df
        self._test_df = test_df
        self._target_col_name = target_col_name

    def review(self):
        title = "Exploring train/test shapes"
        print("=" * 50, title, "=" * 50)
        print("The shape of the train dataset is:", self._train_df.shape)
        print("The shape of the test dataset is:", self._test_df.shape)

        title = "Percentage nans"
        print("=" * 50, title, "=" * 50)
        print(
            "Train: ",
            (100 * self._train_df.isna().sum() / self._train_df.shape[0]).to_dict(),
        )
        print(
            "Test: ",
            (100 * self._test_df.isna().sum() / self._test_df.shape[0]).to_dict(),
        )

        combined_df = pd.concat([self._train_df, self._test_df], keys=["train", "test"])

        title = "Duplicated columns"
        print("=" * 50, title, "=" * 50)
        print(sum(combined_df.drop(columns=self._target_col_name).duplicated()))

        title = "Datatypes"
        print("=" * 50, title, "=" * 50)
        print(combined_df.dtypes.to_dict())

        title = "Numerical columns"
        print("=" * 50, title, "=" * 50)
        numerical_columns = combined_df.select_dtypes(
            include=["number"]
        ).columns.tolist()
        numerical_columns.remove(self._target_col_name)
        self._plot_numerical(numerical_columns)

        title = "Categorical columns"
        print("=" * 50, title, "=" * 50)
        categorical_columns = combined_df.select_dtypes(
            include=["object"]
        ).columns.tolist()
        self._plot_categorical(categorical_columns)

        title = "Linear correlation plot"
        print("=" * 50, title, "=" * 50)
        ordinal_df = combined_df.copy()
        ordinal_encoder = OrdinalEncoder(
            encoded_missing_value=-1,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        ordinal_df[categorical_columns] = ordinal_encoder.fit_transform(
            ordinal_df[categorical_columns]
        )
        correlation_matrix = ordinal_df.corr(method="spearman")

        plt.figure(figsize=(15, 15))
        sns.heatmap(
            correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True
        )
        plt.title("Correlation Heatmap")
        plt.show()

        title = "Target distribution"
        print("=" * 50, title, "=" * 50)
        plt.figure(figsize=(8, 6))
        plt.boxplot(
            combined_df[self._target_col_name], vert=True, patch_artist=True
        )  # vert=True for vertical boxplot
        plt.title(f"Distribution of {self._target_col_name}")
        plt.ylabel(self._target_col_name)
        plt.show()

    def _plot_categorical(self, categorical_columns):
        num_cats = len(categorical_columns)
        num_rows = (num_cats + 2) // 3

        fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
        axes = axes.flatten()

        for i, col in enumerate(categorical_columns):
            train_counts = self._train_df[col].value_counts().to_dict()
            test_counts = self._test_df[col].value_counts().to_dict()
            axes[i].barh(
                [
                    val if len(val) < 7 else val[:4] + "..."
                    for val in train_counts.keys()
                ],
                list(train_counts.values()),
                label="Train dataset",
            )
            axes[i].barh(
                [
                    val if len(val) < 7 else val[:4] + "..."
                    for val in test_counts.keys()
                ],
                list(test_counts.values()),
                label="Test dataset",
            )
            axes[i].set_title(f"Distribution of {col}")
            axes[i].set_xlabel("Count")
            axes[i].set_ylabel(col)
            axes[i].legend()
            axes[i].set_yticklabels([])

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    def _plot_numerical(self, numerical_columns):
        num_cols = len(numerical_columns)
        num_rows = (num_cols + 2) // 3

        fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
        axes = axes.flatten()

        for i, col in enumerate(numerical_columns):
            axes[i].scatter(
                self._train_df[col],
                self._train_df[self._target_col_name],
                alpha=0.5,
                color="blue",
            )
            # axes[i].scatter(self._test_df[col], self._test_df[self._target_col_name], alpha=0.5, color="orange")
            axes[i].set_title(f"{col} vs {self._target_col_name}")
            axes[i].set_xlabel(col)
            axes[i].set_ylabel(self._target_col_name)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()
