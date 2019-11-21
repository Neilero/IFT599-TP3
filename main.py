import pandas as pd
import numpy as np
from kmodes.kmodes import KModes    #https://github.com/nicodv/kmodes/

colNames = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship",
            "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "target"]

def importData(file):
    return pd.read_csv(file, names=colNames)

def main():
    train = importData("adult.data")
    test = importData("adult.test")

    for k in range(2, 10):
        print("--- Computing clustering with {} clusters ---\n".format(k))

        kmodes = KModes(n_clusters=k, verbose=False, n_jobs=-1) #https://github.com/nicodv/kmodes/blob/master/kmodes/kmodes.py#L294
        kmodes.fit(train.drop(columns="target"))

        # Print cluster centroids of the trained model.
        print('Centroids:')
        print(kmodes.cluster_centroids_)
        # Print training statistics
        print('Final training cost: {}'.format(kmodes.cost_))
        print('Training iterations: {}'.format(kmodes.n_iter_))

        clusterIDs, counts = np.unique(kmodes.labels_, return_counts=True)
        counts = [[count] for count in counts]
        clustersStats = dict(zip(clusterIDs, counts))
        #TODO for each cluster:
        #   count each class elements
        #   compute precision and the other one...
        #   (... ?)
        #   print results


if __name__ == '__main__':
    main()