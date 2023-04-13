import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def main() -> None:
    df = pd.read_csv('./../data/processed/happy.csv')
    X = df[['gdp', 'family', 'lifexp', 'freedom',
            'generosity', 'corruption', 'dystopia']]
    y = df['score']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    model = LinearRegression().fit(X_train, y_train)
    y_pred_linear = model.predict(X_test)

    lasso = Lasso(alpha=0.1).fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)

    ridge = Ridge(alpha=0.1).fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)

    linear_loss = mean_squared_error(y_test, y_pred_linear)
    lasso_loss = mean_squared_error(y_test, y_pred_lasso)
    ridge_loss = mean_squared_error(y_test, y_pred_ridge)

    print(f'Linear loss: {linear_loss}')
    print(f'Lasso loss: {lasso_loss}')
    print(f'Ridge loss: {ridge_loss}')


if __name__ == '__main__':
    main()
