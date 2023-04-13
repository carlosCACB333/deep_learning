import pandas as pd
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt


def main() -> None:
    df = pd.read_csv('./../data/processed/candy.csv')
    X = df.drop('competitorname', axis=1)

    kmeans = MiniBatchKMeans(n_clusters=4, random_state=42, batch_size=8)
    kmeans.fit(X)
    print("="*80)
    print(len(kmeans.cluster_centers_))
    df['cluster'] = kmeans.predict(X)

    # plot the clusters
    plt.scatter(df['sugarpercent'], df['pricepercent'], c=df['cluster'])
    plt.show()


if __name__ == '__main__':
    main()
