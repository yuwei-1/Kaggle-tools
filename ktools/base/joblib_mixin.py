from pathlib import Path
from typing import Self
import joblib


class JoblibSaveMixin:
    model = None

    def __init__(self) -> None:
        pass

    def save(self, save_path: str) -> None:
        joblib.dump(self.model, save_path)

    def load(self, load_path: str):
        self.model = joblib.load(load_path)
        return self


class ArtifactSaveMixin:
    name: str = "artifact-name"

    def save(self, dir_path: str) -> None:
        save_path = Path(dir_path) / (self.name + ".pkl")
        joblib.dump(self, save_path)

    @classmethod
    def load(cls, dir_path: str) -> Self:
        load_path = Path(dir_path) / (cls.name + ".pkl")
        return joblib.load(load_path)
