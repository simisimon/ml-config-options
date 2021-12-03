from sklearn.cluster import Birch
X = [[0, 1], [0.3, 1], [-0.3, 1], [0, -1], [0.3, -1], [-0.3, -1]]
brc = Birch(n_clusters=None)
brc.fit(X)
brc.predict(X)

birch_models = [
    Birch(1.7, n_clusters=None),
    Birch(threshold=1.7, n_clusters=100),
]