from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings
import sklearn.ensemble
from sklearn.model_selection import GridSearchCV

bin_cols = ["is_male"]

logistic_reg = LogisticRegression(
    multi_class="multinomial", solver="lbfgs", C=5
)

rnd_forest = RandomForestClassifier(100, 2)

mlp = MLPClassifier(hidden_layer_sizes=(13, 13))

mdl = []
mdl.append(SVC(class_weight=None, probability=True))

pre = ColumnTransformer(
    [
        ("OneHotEncoder", OneHotEncoder(drop=bin_cols)),
        ("Scale", StandardScaler()),
    ],
    remainder="passthrough",
)

grid = {
    "Model__n_estimators": [100, 150],
    "Model__max_depth": [3, 4, 5],
}

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