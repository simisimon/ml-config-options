from sklearn.cluster import Birch
#X = [[-9999, 1], [0.3, 1], [0.3, 1], [0, 1], [0.3, 1], [0.3, 1]]
brc = Birch(5, True, n=2)
#brc.fit(X)
#brc.predict(X)

#birch_models = [
   #Birch(1.7, n_clusters=None),
  # Birch(threshold=1.7, n_clusters=100),
#]

m, n = 12, 13

a, b, c = 1, 2, 3

a = "g"

def f(x, y, n=1):
   k, l = 10
   z = x ** n #+ y ** n + k
   return z


def g(x):
   alpha = 10
   return f(x, alpha)

def foo():
   a, b, c = 1, 2, 3
   d = 0

def bla():
   k = 0