import pandas as pd
import numpy as np
from kmodes.kmodes import KModes    #https://github.com/nicodv/kmodes/
from sklearn.cluster import KMeans  #https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

COL_NAMES = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship",
            "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "target"]

def importData():

    train = pd.read_csv("adult.data", names=COL_NAMES, skipinitialspace=True)
    train = prepareData(train)

    test = pd.read_csv("adult.test", names=COL_NAMES, skipinitialspace=True, header=1)
    test = prepareData(test)

    return train, test

def prepareData(data):
    #remove "empty" columns
    data.drop(columns=["capital-gain", "capital-loss", "hours-per-week"], inplace=True)

    #categorize numerical columns
    categorizedData = categorizeData(data)

    return categorizedData

def categorizeData(data):
    data["age"] = categorizeVariable(data["age"], 10)
    data["fnlwgt"] = categorizeVariable(data["fnlwgt"], 4)
    data["education-num"] = categorizeVariable(data["education-num"], 6)

    return data

def categorizeVariable(variable, k_cluster):
    kmean = KMeans(n_clusters=k_cluster, verbose=False, n_jobs=-1)

    reshapedVariable = variable.values.reshape(-1, 1)
    categorizedVariable = kmean.fit_predict(reshapedVariable)
    return categorizedVariable

def main():
    train, test = importData()

    for k in np.unique(np.geomspace(2, 75, 20, dtype=int)):
        print("--- Computing clustering with {} clusters ---\n".format(k))

        # training
        kmodes = KModes(n_clusters=k, verbose=False, n_jobs=-1) #https://github.com/nicodv/kmodes/blob/master/kmodes/kmodes.py#L294
        kmodes.fit(train.drop(columns="target"))

        # testing
        test["prediction"] = kmodes.predict(test.drop(columns="target"))

        # Print training statistics
        print('Final training cost: {}'.format(kmodes.cost_))
        print('Training iterations: {}'.format(kmodes.n_iter_))

        clusterIDs, stats = np.unique(kmodes.labels_, return_counts=True)

        # stat for each cluster = [trainCount, trainCount<=50K, trainCount>50K, clusterClass, trainingError, testCount, testCount<=50K, testCount>50K, testError]
        #                              0              1               2               3             4            5            6               7            8
        stats = np.hstack((np.reshape(stats, (-1,1)), np.zeros((k,8)))).tolist()
        clustersStats = dict(zip(clusterIDs, stats))

        # count training elements
        for i, data in enumerate(train.values):
            if data[-1] == "<=50K":
                clustersStats[kmodes.labels_[i]][1] += 1
            else:
                clustersStats[kmodes.labels_[i]][2] += 1

        # count testing elements
        for data in test[["target", "prediction"]].values:
            clustersStats[data[1]][5] += 1

            if data[0] == "<=50K.":
                clustersStats[data[1]][6] += 1
            else:
                clustersStats[data[1]][7] += 1

        # dropping predictions after use to not influence next iteration
        test.drop(columns="prediction", inplace=True)

        globalTrainError = 0
        globalTestError = 0
        for stats in clustersStats.values():
            stats[3] = "<=50K" if np.argmax(stats[1:3]) == 0 else ">50K"

            if stats[3] == "<=50K":
                stats[4] = stats[2] / stats[0]
                stats[8] = stats[7] / stats[5]
            else:
                stats[4] = stats[1] / stats[0]
                stats[8] = stats[6] / stats[5]

            print("Cluster stats : ", stats)
            globalTrainError += stats[4] * (stats[0] / train.shape[0])
            globalTestError += stats[8] * (stats[5] / test.shape[0])


        print("Global training error : {0:.2f} %".format(globalTrainError*100))
        print("Global testing  error : {0:.2f} %".format(globalTestError*100))

if __name__ == '__main__':
    main()