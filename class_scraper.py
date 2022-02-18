from urllib.request import urlopen
from bs4 import BeautifulSoup
import json


class ClassScraper:
    def scrape_parameters(self, elements):
        parameters = {}
        for e in elements:
            split = e.text.split("=", 1)
            param = split[0]
            if len(split) > 1:
                default_value = split[1]
                parameters[param] = default_value
            else:
                parameters[param] = None
        return parameters

    def create_json(self, library, classes):
        with open("{0}.txt".format(library), 'w') as outfile:
            json.dump(classes, outfile, indent=4)


class SklearnScraper(ClassScraper):
    def __init__(self):
        self.class_urls = []
        self.classes = {}

    def get_classes(self):
        self.scrape_class_urls()
        self.scrape_classes()
        self.create_json("sklearn", self.classes)

    def scrape_class_urls(self):
        link = "https://scikit-learn.org/stable/modules/classes.html#"
        html = urlopen(link)
        soup = BeautifulSoup(html, "html.parser")

        table_rows = soup.body.findAll("tr")
        for row in table_rows:
            text = row.find("a").text
            text = text[text.rfind(".") + 1:]
            if text[0].isupper():
                url = row.find("a").attrs["href"]
                self.class_urls.append(url)

    def scrape_classes(self):
        for url in self.class_urls:
            link = "https://scikit-learn.org/stable/modules/" + url
            html = urlopen(link)
            soup = BeautifulSoup(html, "html.parser")

            data_table = soup.find("dt", {"class": "sig sig-object py"})
            elements = data_table.findAll("em", {"class": "sig-param"})
            class_ = data_table.find("span", {"class": "sig-name descname"}).text

            parameters = self.scrape_parameters(elements)
            self.classes[class_] = parameters


class TorchScraper(ClassScraper):
    def __init__(self):
        self.module_urls = []
        self.class_urls = []
        self.desc_elements = []
        self.classes = {}

    def get_classes(self):
        self.scrape_module_urls()
        self.scrape_class_urls()
        self.scrape_desc_elements()
        self.scrape_classes()
        self.create_json("pytorch", self.classes)

    def scrape_module_urls(self):
        link = "https://pytorch.org/docs/stable/index.html"
        html = urlopen(link)
        soup = BeautifulSoup(html, "html.parser")

        caption = soup.find("p", {"class": "caption"}, text="Python API")
        ul = caption.find_next("ul")
        li = ul.findAll("li", {"class": "toctree-l1"})

        for element in li:
            url = element.find("a").attrs["href"]
            self.module_urls.append(url)

    def scrape_class_urls(self):
        for url in self.module_urls:
            try:
                link = "https://pytorch.org/docs/stable/" + url
                html = urlopen(link)
                soup = BeautifulSoup(html, "html.parser")

                if url == "distributed.elastic.html":
                    caption = soup.find("p", {"class": "caption"}, text="API")
                    ul = caption.find_next("ul")
                    li = ul.findAll("li", {"class": "toctree-l1"})
                    for element in li:
                        url = element.find("a").attrs["href"]
                        self.class_urls.append(url)

                else:
                    table_rows = soup.body.findAll("tr")
                    for row in table_rows:
                        a = row.find("a")
                        if a is not None and "title" in a.attrs:
                            title = a.attrs["title"]
                            title = title[title.rfind(".") + 1:]
                            if title[0].isupper():
                                url = row.find("a").attrs["href"]
                                self.class_urls.append(url)

                    dl = soup.findAll("dl", {"class": "py class"})
                    self.desc_elements.extend(dl)
            except:
                continue

    def scrape_desc_elements(self):
        for url in self.class_urls:
            link = "https://pytorch.org/docs/stable/" + url
            html = urlopen(link)
            soup = BeautifulSoup(html, "html.parser")
            dl = soup.findAll("dl", {"class": "py class"})
            if len(dl) > 0:
                for element in dl:
                    self.desc_elements.append(element)

    def scrape_classes(self):
        for desc_element in self.desc_elements:
            dt = desc_element.find("dt")
            if "id" in dt.attrs:
                class_ = dt.attrs["id"]
                class_ = class_[class_.rfind(".") + 1:]
                elements = desc_element.findAll("em", {"class": "sig-param"})
                parameters = self.scrape_parameters(elements)
                self.classes[class_] = parameters


def main():
    SklearnScraper().get_classes()
    TorchScraper().get_classes()


if __name__ == "__main__":
    main()
