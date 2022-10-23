import numpy as np

# each column represents a distance metric, and rows represent the distance of documents within the same news-group
# and between different newsgroups.
means = np.array([[10.52, 86.36, 1.30, 0.44, 0.72, 0.57],
                  [10.24, 82.38, 1.23, 0.39, 0.67, 0.52],
                  [13.04, 128.16, 1.32, 0.39, 0.69, 0.53],
                  [11.34, 94.88, 1.33, 0.40, 0.72, 0.56],
                  [11.88, 134.42, 1.27, 0.42, 0.69, 0.54],
                  [10.09, 80.38, 1.18, 0.33, 0.62, 0.47],
                  [12.79, 125.02, 1.27, 0.33, 0.64, 0.48],
                  [11.18, 92.60, 1.28, 0.35, 0.68, 0.52],
                  [11.71, 132.66, 1.22, 0.37, 0.64, 0.49],
                  [15.28, 165.33, 1.33, 0.33, 0.65, 0.48],
                  [13.85, 136.74, 1.35, 0.34, 0.69, 0.52],
                  [14.32, 173.86, 1.30, 0.37, 0.65, 0.50],
                  [12.20, 104.91, 1.37, 0.36, 0.72, 0.55],
                  [12.68, 143.52, 1.31, 0.38, 0.69, 0.53],
                  [12.97, 179.84, 1.24, 0.39, 0.65, 0.50]], dtype=np.float64)

# each column represents a distance metric, and rows represent the distance of documents within the same news-group
intra = np.array([[10.52, 86.36, 1.30, 0.44, 0.72, 0.57],
                  [10.09, 80.38, 1.18, 0.33, 0.62, 0.47],
                  [13.85, 136.74, 1.35, 0.34, 0.69, 0.52],
                  [12.20, 104.91, 1.37, 0.36, 0.72, 0.55]], dtype=np.float64)
# each column represents a distance metric, and rows represent the distance of documents between different newsgroups.
inter = np.array([[10.24, 82.38, 1.23, 0.39, 0.67, 0.52],
                    [11.18, 92.60, 1.28, 0.35, 0.68, 0.52],
                    [12.79, 125.02, 1.27, 0.33, 0.64, 0.48],
                    [11.88, 134.42, 1.27, 0.42, 0.69, 0.54],
                    [11.71, 132.66, 1.22, 0.37, 0.64, 0.49],
                    [12.68, 143.52, 1.31, 0.38, 0.69, 0.53],
                    [13.04, 128.16, 1.32, 0.39, 0.69, 0.53],
                    [14.32, 173.86, 1.30, 0.37, 0.65, 0.50],
                    [12.97, 179.84, 1.24, 0.39, 0.65, 0.50],
                    [15.28, 165.33, 1.33, 0.33, 0.65, 0.48]], dtype=np.float64)

print('distance mean', means.mean(axis=0))
print('distance stddev', means.std(axis=0))
print('distance var', means.var(axis=0))
print('index of minimum per distance', means.argmin(axis=0))
print('index of maximum per distance', means.argmax(axis=0))

print('intra mean', intra.mean(axis=0))
print('intra stddev', intra.std(axis=0))
print('intra var', intra.var(axis=0))
print('index of minimum per intra', intra.argmin(axis=0))
print('index of maximum per intra', intra.argmax(axis=0))

print('inter mean', inter.mean(axis=0))
print('inter stddev', inter.std(axis=0))
print('inter var', inter.var(axis=0))
print('index of minimum per inter', inter.argmin(axis=0))
print('index of maximum per inter', inter.argmax(axis=0))

print('group with smallest intra-group distance', intra.mean(axis=1).argmin())
print('group with largest intra-group distance', intra.mean(axis=1).argmax())
print('group with smallest inter-group distance', inter.mean(axis=1).argmin())
print('group with largest inter-group distance', inter.mean(axis=1).argmax())

