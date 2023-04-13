import pandas as pd
import numpy as np

from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from utils.util import Util


class Models:

    def __init__(self) -> None:
        self.reg = {
            'SVR': SVR(),
            'GRADIENT': GradientBoostingRegressor()
        }

        self.params = {
            'SVR': {
                'kernel': ['linear', 'poly', 'rbf'],
                'gamma': ['auto', 'scale'],
                'C': [1, 5, 10]
            }, 'GRADIENT': {
               'loss': ['squared_error', 'huber', 'quantile'],
               'learning_rate': [0.01, 0.05, 0.1]
            }
        }

    def grid_train(self, X, y) -> None:

        best_score = 999
        best_model = None

        for name, reg in self.reg.items():

            grid_reg = GridSearchCV(
                reg, self.params[name], cv=3).fit(X, y.values.ravel())
            score = np.abs(grid_reg.best_score_)

            if score < best_score:
                best_score = score
                best_model = grid_reg.best_estimator_

        utils = Util()
        utils.model_export(best_model, 'models/model.pkl')
        print(f'Score: {best_score}')
