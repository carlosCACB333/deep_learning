import pandas as pd
import pathlib
import joblib


class Util:
    root_dir = pathlib.Path(__file__).parent.parent

    def load_from_csv(self, path: str) -> pd.DataFrame:
        return pd.read_csv(self.root_dir / path)

    def X_y_split(self, df: pd.DataFrame, y_col: str) -> tuple:
        return df.drop(y_col, axis=1), df[y_col]

    def model_export(self, model, path: str) -> None:
        joblib.dump(model, self.root_dir / path)
