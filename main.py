from utils.util import Util
from model import Models
if __name__ == '__main__':
    util = Util()
    df = util.load_from_csv('data/processed/happy.csv')
    df.drop('country', axis=1, inplace=True)
    df.drop('rank', axis=1, inplace=True)
    X, y = util.X_y_split(df, 'score')
    model = Models()
    model.grid_train(X, y)
