import joblib


class JoblibSaverMixin():

    model = None

    def __init__(self) -> None:
        pass

    def save(self, save_path : str) -> None:
        joblib.dump(self.model, save_path)
    
    def load(self, load_path : str):
        self.model = joblib.load(load_path)
        return self