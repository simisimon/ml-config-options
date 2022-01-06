from urllib.request import urlopen
from bs4 import BeautifulSoup
from pprint import pprint


def get_sklearn_classes():
    link = "https://scikit-learn.org/stable/modules/classes.html#"
    html = urlopen(link)
    soup = BeautifulSoup(html, "html.parser")
    table_rows = soup.body.findAll("tr")
    sklearn_objects = []
    for o in table_rows:
        obj = o.find("a").text
        sklearn_objects.append(obj)

    sklearn_classes = []
    for s in sklearn_objects:
        index = [i for i, c in enumerate(s) if c.isupper()]
        for i in index:
            if s[i-1] == ".":
                sklearn_classes.append(s[i:])
                break

    return sklearn_classes

sklearn_classes = get_sklearn_classes()
#pprint(sklearn_classes)