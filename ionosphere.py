import numpy as np

data = np.genfromtxt('dataset/ionosphere/ionosphere.data', delimiter=',', dtype=None, encoding='utf-8', converters={34: lambda x: 1 if x == 'g' else 0})
np.random.shuffle(data)
trainLen = int(len(data) * 0.8)
train, test = data[:trainLen], data[trainLen:]
trainResult = [i[-1] for i in train]
train = [list(t)[:-1] for t in train]
testResult = [i[-1] for i in test]
test = [list(t)[:-1] for t in test]
