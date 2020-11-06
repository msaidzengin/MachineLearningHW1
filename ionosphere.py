import numpy as np

data = np.genfromtxt('ionosphere/ionosphere.data', delimiter=',', dtype=None, encoding='utf-8')
np.random.shuffle(data)
trainLen = int(len(data) * 0.8)
train, test = data[:trainLen], data[trainLen:]

print(len(data))
print(len(train))
print(len(test))

