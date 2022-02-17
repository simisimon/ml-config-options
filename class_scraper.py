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


def get_torch_module_urls():
    link = "https://pytorch.org/docs/stable/index.html"
    html = urlopen(link)
    soup = BeautifulSoup(html, "html.parser")

    caption = soup.find("p", {"class": "caption"}, text="Python API")
    ul = caption.find_next("ul")
    li = ul.findAll("li", {"class": "toctree-l1"})
    module_url = []
    for element in li:
        url = element.find("a").attrs["href"]
        module_url.append(url)

    return module_url


def get_torch_class_urls(urls):
    torch_cls = {}
    cls_urls = []
    for u in urls:
        try:
            url = "https://pytorch.org/docs/stable/" + u
            html = urlopen(url)
            soup = BeautifulSoup(html, "html.parser")

            table_rows = soup.body.findAll("tr")

            for row in table_rows:
                a = row.find("a")
                if a != None and "title" in a.attrs:
                    title = a.attrs["title"]
                    title = title[title.rfind("."):]
                    if title[1].isupper():
                        url = row.find("a").attrs["href"]
                        cls_urls.append(url)

            dl = soup.findAll("dl", {"class": "py class"})
            if len(dl) > 0:
                for d in dl:
                    dt = d.find("dt")
                    if "id" in dt.attrs:
                        cls = dt.attrs["id"]
                        cls = cls[cls.rfind(".") + 1:]
                        elements = d.findAll("em", {"class": "sig-param"})
                        param = {}
                        for ele in elements:
                            split = ele.text.split("=", 1)
                            p = split[0]
                            if len(split) > 1:
                                default_value = split[1]
                                param[p] = default_value
                            else:
                                param[p] = None
                        torch_cls[cls] = param
        except:
            continue
    return 0

def create_json(classes):
    with open("sklearn.txt", 'w') as outfile:
        json.dump(classes, outfile, indent=4)


def main():
    #sklearn_class_urls = get_sklearn_class_urls()
    #sklearn_classes = get_sklearn_classes(sklearn_class_urls)
    #create_json(sklearn_classes)
    torch_module_urls = get_torch_module_urls()
    torch_class_urls = get_torch_class_urls(torch_module_urls)



if __name__ == "__main__":
    main()



