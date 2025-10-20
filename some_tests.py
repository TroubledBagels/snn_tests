import numpy as np
all_classes = [[1,2,3], [4,5], [6,7,8,9], [10]]
dist = np.array([len(c) for c in all_classes])
dist = dist / dist.sum()
print(dist)
