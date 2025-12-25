import yaml
from typing import Dict, Callable, Any
from optuna.trial import Trial


NestedDict = dict[str, "NestedDict | Any"]
TrialSampler = Callable[[Trial], Any]


def load_optuna_grid(
    path: str,
    model_type: str,
    extra_samplers: Dict[str, TrialSampler] | None = None,
) -> Callable[[Trial], Dict[str, Any]]:
    """
    Load an Optuna parameter grid from a YAML file.

    Args:
        path: Path to the YAML file containing parameter grids.
        model_type: Key in the YAML file for the model's parameter grid.
        extra_samplers: Additional tunable parameters as Optuna callables.
            Each callable takes a Trial and returns a sampled value.
            Example: {"weight": lambda t: t.suggest_float("weight", 0.5, 2.0)}

    Returns:
        A callable that takes an Optuna Trial and returns sampled parameters.
    """
    with open(path, "r") as f:
        param_grid: NestedDict = yaml.safe_load(f)
    param_grid = param_grid.get(model_type, {})
    if len(param_grid) == 0:
        raise ValueError(f"No parameter grid found for model type: {model_type}")

    def param_grid_getter(trial: Trial) -> Dict[str, Any]:
        unpacked = {}
        for param_name, param_info in param_grid.items():
            dtype = param_info.get("type")
            if dtype == "int":
                unpacked[param_name] = trial.suggest_int(
                    param_name,
                    param_info["low"],
                    param_info["high"],
                )
            elif dtype == "float":
                unpacked[param_name] = trial.suggest_float(
                    param_name,
                    param_info["low"],
                    param_info["high"],
                    log=param_info.get("log", False),
                )
            elif dtype == "categorical":
                unpacked[param_name] = trial.suggest_categorical(
                    param_name,
                    param_info["choices"],
                )
            elif dtype == "fixed":
                unpacked[param_name] = trial.suggest_categorical(
                    param_name, [param_info["value"]]
                )
            else:
                raise ValueError(f"Unsupported parameter type: {dtype}")

        if extra_samplers:
            for param_name, sampler in extra_samplers.items():
                unpacked[param_name] = sampler(trial)

        return unpacked

    return param_grid_getter
