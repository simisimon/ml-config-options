import tensorflow as tf
with tf.Graph().as_default():
    q = tf.TensorShape(dims=
        tf.float32, name="Q", shape=tf.Graph([1, 5, 2, 8]))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings
import sklearn.ensemble
from sklearn.model_selection import GridSearchCV

class cls:
    def class_func(self):
        zip = 777
        test_func(1111,2222,3333,zip)

def test2(zum):
    zub = 444
    if x < 2:
        zub = 555
    else:
        zub = 666
    test_func(111, 222, 333, zub)

x = hallo(test_func(1, 2, a, xyz, 8, 9))

#def test_func(a= 0, b=1, c = 2, d = 3, *e, f, x = 4 ):
class cls_test2:
    def bla(self):
        self.test_func(101, 102, 103, 104,105 ,106)

    def test_func(self, y, u, zzz = 3, /, x =5, *z, c = 8, j = 2, **v):
        a = 2
        #x = "a"
        bin_cols = ["is_male"]

        if x == 2:
            bin_cols = ["is_female"]
        else:
            bin_cols = ["is_nonbinary"]

        logistic_reg = LogisticRegression(
            multi_class="multinomial", solver="lbfgs", C=5
        )

        rnd_forest = RandomForestClassifier(100, 2)

        mlp = MLPClassifier(hidden_layer_sizes=(13, 13))

        mdl = []
        mdl.append(SVC(class_weight=None, probability=True))

        pre = ColumnTransformer(
            [
                ("OneHotEncoder", OneHotEncoder(drop=x)),
                ("Scale", StandardScaler()),
            ],
            remainder="passthrough",
        )

        x = "hiiii"

grid = {
    "Model__n_estimators": [100, 150],
    "Model__max_depth": [3, 4, 5],
}

x = 2
def test2():
    #test_func(x = 3)
    #test_func(x = j)
    x =2

with warnings.catch_warnings():
    grid_SCV = sklearn.model_selection.GridSearchCV(pre, grid)

for i in range(10):
    try:
        svm = sklearn.svm.LinearSVC()
    except Exception:
        svm = None


def available_cpu_count():
    return 1


rf = sklearn.ensemble.RandomForestRegressor(n_jobs=available_cpu_count())

onehotencoder = OneHotEncoder(categorical_features=[0])

grid_search = GridSearchCV(n_jobs=-1)
