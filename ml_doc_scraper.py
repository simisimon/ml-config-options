from inspect import getmembers, isfunction
from urllib.request import urlopen

import bs4
from bs4 import BeautifulSoup
from pprint import pprint

import sklearn, mlflow, tensorflow as tf #, torch

def get_dir(ml_lib):
    lib_dir = dir(ml_lib)
    pprint.pprint(lib_dir)
    lib_dir = [x for x in lib_dir if not x.startswith('_')]
    return lib_dir

def get_func(ml_lib):
    func_tuples = getmembers(ml_lib, isfunction)
    func = [f[0] for f in func_tuples]
    return func

def get_tf_modules():
    link = "https://www.tensorflow.org/api_docs/python/tf"
    html = urlopen(link)
    soup = BeautifulSoup(html, "html.parser")

    modules = ["tf"]

    for header in soup.find_all("h2", {"id": "modules_2"}):
        nextNode = header
        while True:
            nextNode = nextNode.nextSibling
            if nextNode is None:
                break
            if isinstance(nextNode, bs4.Tag):
                if nextNode.name == "h2":
                    break
                module = nextNode.get_text(strip=True).strip().split(":")[0]
                modules.append("tf." + module)

    return modules

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
#pprint(sklearn_func)
#pprint(sklearn_modules)

#torch_func = get_func(torch) #not working sufficiently
#pprint(torch_func)

mlflow_func = get_func(mlflow)
mlflow_modules = get_mlflow_modules()
#pprint(mlflow_func)
#pprint(mlflow_modules)

tf_func = get_func(tf)
tf_modules = get_tf_modules()
#pprint(tf_modules)
#pprint(tf_func)

