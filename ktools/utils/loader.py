import yaml
from typing import Dict, Callable, Any
from optuna.trial import Trial


NestedDict = dict[str, "NestedDict | Any"]


def load_optuna_grid(path: str, model_type: str) -> Callable:
    unpacked = {}
    with open(path, "r") as f:
        param_grid: NestedDict = yaml.safe_load(f)
    param_grid = param_grid.get(model_type, {})
    if len(param_grid) == 0:
        raise ValueError(f"No parameter grid found for model type: {model_type}")

    def param_grid_getter(trial: Trial) -> Dict:
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
                unpacked[param_name] = trial.set_user_attr(
                    param_name, param_info["value"]
                )
            else:
                raise ValueError(f"Unsupported parameter type: {dtype}")
        return unpacked

    return param_grid_getter
