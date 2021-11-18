from inspect import getmembers, isfunction
from urllib.request import urlopen
from bs4 import BeautifulSoup
from pprint import pprint

import sklearn, mlflow #torch #tensorflow

def get_dir(ml_lib):
    lib_dir = dir(ml_lib)
    pprint.pprint(lib_dir)
    lib_dir = [x for x in lib_dir if not x.startswith('_')]
    return lib_dir

def get_func(ml_lib):
    func_tuples = getmembers(ml_lib, isfunction)
    func = [f[0] for f in func_tuples]
    return func

def get_sklearn_modules():
    link = "https://scikit-learn.org/stable/modules/classes.html#"
    html = urlopen(link)
    soup = BeautifulSoup(html, "html.parser")
    headers = soup.body.findAll("h2")

    modules = ['sklearn']
    for h in headers:
        a = h.find('a')
        modules.append(a.attrs['href'][8:])

    modules.pop()
    return modules

def get_mlflow_modules():
    link = "https://mlflow.org/docs/latest/python_api/index.html"
    html = urlopen(link)
    soup = BeautifulSoup(html, "html.parser")
    mlflow_list = soup.find("li", {"class": "current"})

    modules = []
    for m in mlflow_list.find_all("li"):
        a = m.find('a')
        modules.append(a.text)

    return modules

sklearn_func = get_func(sklearn)
sklearn_modules = get_sklearn_modules()
#print(sklearn_func)
#print(sklearn_modules)

#torch_func = get_func(torch) #not working sufficiently

mlflow_func = get_func(mlflow)
mlflow_modules = get_mlflow_modules()
#pprint(mlflow_func)
#pprint(mlflow_modules)






