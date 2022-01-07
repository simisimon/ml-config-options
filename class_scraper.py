from urllib.request import urlopen
from bs4 import BeautifulSoup
import json


def get_sklearn_class_urls():
    link = "https://scikit-learn.org/stable/modules/classes.html#"
    html = urlopen(link)
    soup = BeautifulSoup(html, "html.parser")
    table_rows = soup.body.findAll("tr")

    class_urls = []
    for t in table_rows:
        obj = t.find("a").text
        index = [i for i, c in enumerate(obj) if c.isupper()]
        for i in index:
            if obj[i-1] == ".":
                url = t.find("a").attrs["href"]
                class_urls.append(url)
                break

    return class_urls


def get_sklearn_classes(urls):
    sklearn_classes = {}
    for u in urls:
        url = "https://scikit-learn.org/stable/modules/" + u
        html = urlopen(url)
        soup = BeautifulSoup(html, "html.parser")

        data_table = soup.find("dt", {"class": "sig sig-object py"})
        elements = data_table.findAll("em", {"class": "sig-param"})
        class_name = data_table.find("span", {"class": "sig-name descname"}).text
        parameter = {}
        for e in elements:
            split = e.text.split("=", 1)
            p = split[0]
            if len(split) > 1:
                default_value = split[1]
                parameter[p] = default_value
            else:
                parameter[p] = None

        sklearn_classes[class_name] = parameter

    return sklearn_classes


def create_json(classes):
    with open("sklearn.txt", 'w') as outfile:
        json.dump(classes, outfile, indent=4)


def main():
    sklearn_class_urls = get_sklearn_class_urls()
    sklearn_classes = get_sklearn_classes(sklearn_class_urls)
    create_json(sklearn_classes)


if __name__ == "__main__":
    main()



