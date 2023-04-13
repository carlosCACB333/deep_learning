import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def main() -> None:
    df = pd.read_csv('./../data/processed/heart.csv')
    X = df.drop('target', axis=1)
    y = df['target']

    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    pca = PCA(n_components=3)
    pca.fit(X_train)

    # plt.plot(range(len(pca.explained_variance_)),
    #          pca.explained_variance_ratio_)
    # plt.show()

    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    logistic = LogisticRegression()
    logistic.fit(X_train, y_train)
    print(logistic.score(X_test, y_test))


if __name__ == '__main__':
    main()
