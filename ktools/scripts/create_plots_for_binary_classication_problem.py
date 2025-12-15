import ast
import argparse
from pathlib import Path
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_object_dtype
from ktools.visualization.classification import (
    plot_professional_violin,
    plot_professional_boxplot,
    plot_cross_tabulation,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create plots for binary classification problem."
    )
    parser.add_argument("data_file", type=str, help="Path to the .csv data file.")
    parser.add_argument("target_column", type=str, help="Name of the target column.")
    parser.add_argument(
        "--mapping", type=str, help="Optional mapping for target values."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/plots",
        help="Directory to save the plots.",
    )
    args = parser.parse_args()

    data_file: str = args.data_file
    target_column: str = args.target_column
    mapping: str = args.mapping
    output_dir = Path(args.output_dir)

    mapping_dict = ast.literal_eval(mapping) if mapping else None

    data = pd.read_csv(data_file)
    target_values = (
        data[target_column]
        if mapping_dict is None
        else data[target_column].map(mapping_dict)
    )
    features = [col for col in data.columns if col != target_column]

    for feature in features:
        feat_values = data[feature]

        if is_numeric_dtype(feat_values):
            plot_professional_violin(
                categories=target_values,
                values=feat_values,
            ).write_image(output_dir / f"{feature}_violin_plot.png")

            plot_professional_boxplot(
                categories=target_values,
                values=feat_values,
            ).write_image(output_dir / f"{feature}_boxplot.png")

        elif is_object_dtype(feat_values):
            plot_cross_tabulation(
                categories=target_values,
                values=feat_values,
            ).write_image(output_dir / f"{feature}_cross_tabulation.png")
        else:
            raise NotImplementedError(
                f"Feature type of column '{feature}' is not supported."
            )
