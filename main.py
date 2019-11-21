import pandas as pd
import numpy as np
from kmodes.kmodes import KModes    #https://github.com/nicodv/kmodes/

COL_NAMES = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship",
            "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "target"]
dataStats = None    # will be computed in main
GROUPS_NUMBER = 20
GROUPS_NUMBER_EDUCATIONAL = 10


def importData(file):
    return pd.read_csv(file, names=COL_NAMES, skipinitialspace=True)

def categorizeVariable(variable, value, groupNumber):
    max = dataStats[variable]["max"]
    min = dataStats[variable]["min"]

    groupSize = np.floor((max-min)/groupNumber)

    groupValue = (value//groupSize) * groupSize

    return "{}-{}".format(groupValue, groupValue+groupSize)


def categorizeAge(value):
    return categorizeVariable("age", value, GROUPS_NUMBER)

def categorizeFnlwgt(value):
    return categorizeVariable("fnlwgt", value, GROUPS_NUMBER)

def categorizeEducationNum(value):
    return categorizeVariable("education-num", value, GROUPS_NUMBER_EDUCATIONAL)

def categorizeCapitalGain(value):
    return categorizeVariable("capital-gain", value, GROUPS_NUMBER)

def categorizeCapitalLoss(value):
    return categorizeVariable("capital-loss", value, GROUPS_NUMBER)

def categorizeHours(value):
    return categorizeVariable("hours-per-week", value, GROUPS_NUMBER)

def categorizeData(data):
    # TODO think about it in all cases :
    #   - use K-mean (like Johana) ?    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    #   - Some vars might be better off removing ?
    #   ...

    data["age"] = data["age"].apply(categorizeAge)
    data["fnlwgt"] = data["fnlwgt"].apply(categorizeFnlwgt)
    data["education-num"] = data["education-num"].apply(categorizeEducationNum)
    data["capital-gain"] = data["capital-gain"].apply(categorizeCapitalGain)
    data["capital-loss"] = data["capital-loss"].apply(categorizeCapitalLoss)
    data["hours-per-week"] = data["hours-per-week"].apply(categorizeHours)

    return data

def main():
    global dataStats
    train = importData("adult.data")
    test = importData("adult.test")

    dataStats = train.describe()
    train = categorizeData(train)

    for k in range(15, 16):
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
        stats = np.hstack((stats, np.zeros((k,8))))
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

            if data[0] == "<=50K":
                clustersStats[data[1]][6] += 1
            else:
                clustersStats[data[1]][7] += 1

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

            print(stats)
            globalTrainError += stats[4] * (stats[0] / train.shape[0])
            globalTestError += stats[8] * (stats[5] / test.shape[0])


        print("Global training error : {0:.2f} %".format(globalTrainError*100))
        print("Global testing  error : {0:.2f} %".format(globalTestError*100))

if __name__ == '__main__':
    main()