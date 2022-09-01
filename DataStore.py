import os
import glob
import numpy as np
import random
import pandas as pd
MIN = 25
MAX = 55
def str2ary(str_):
    x = str_.replace("]","").replace("[","")
    x = x.split(",")
    assert(not len(x)%2)
    lst = []
    for i in range(0, len(x), 2):
        tmp = range(int(x[i]), int(x[i+1]))
        lst.append(tmp)
    return lst

class XY():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.labels = []
    def shuffle(self):
        assert(len(self.x) == len(self.y))
        for i in len(self.x):
            replace_i = random.randint(0, len(self.x))
            tmp = self.x[i]
            self.x[i] = self.x[replace_i]
            self.x[replace_i] = tmp

            tmp = self.y[i]
            self.y[i] = self.y[replace_i]
            self.y[replace_i] = tmp


class DataStore():
    def __init__(self, dataDir: os.PathLike, seqLength: int):
        self.seqLength = seqLength
        self.trainData = {}
        self.testData = {}
        self.__load(dataDir)

    def __load(self, datadir: os.PathLike):
        glb = glob.glob(os.path.join(datadir, "train", "*"))
        for g in glb:
            self.trainData[g.split("/")[-1].split(".")[0]] = self.__preprocess(np.load(g))

        glb = glob.glob(os.path.join(datadir, "test", "*"))
        for g in glb:
            self.testData[g.split("/")[-1].split(".")[0]] = self.__preprocess(np.load(g))

        for rw in pd.read_csv(os.path.join(datadir, "labeled_anomalies.csv")):
            self.testData[rw[1]['chan_id']].labels = str2ary(rw[1]['anomaly_sequences'])

    def __preprocess(self, sample: np.array):
        xs = []
        ys = []
        for i in range(sample.shape[0]):
            if i + self.seqLength + 1 >= sample.shape[0]:  # X chunk + Y sequence
                break
            if sample.shape[1] == MIN:
                x = np.concatenate((sample[i:i+self.seqLength], \
                                    np.zeros((self.seqLength, MAX - MIN))), axis=1)
                y = np.concatenate(([sample[i + self.seqLength]], \
                                    np.zeros((1, MAX - MIN), )), axis=1)
            else:
                x = sample[i:i+self.seqLength]
                y = sample[i + self.seqLength]
            xs.append(x.reshape((MAX, self.seqLength)))
            ys.append(y.reshape((MAX, 1)))
        return XY(xs, ys)
