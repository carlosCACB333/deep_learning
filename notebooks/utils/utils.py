
from sklearn.model_selection import train_test_split


def tran_val_test_split(df, target, test_size=0.3, random_state=42, shuffle=True, stratify=None):

    X = df.drop(target, axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=shuffle, stratify=stratify)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=.5, random_state=random_state, shuffle=shuffle, stratify=stratify)

    return X_train, X_val, X_test, y_train, y_val, y_test
