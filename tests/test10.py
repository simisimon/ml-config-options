from sklearn import BaseEstimator

class C(BaseEstimator):
    def __init__(self, a, b):
        self.a = a
        self.b = b

d = 1

def func(e):
    C(e, 5)

func(d)

