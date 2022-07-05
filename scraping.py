from urllib.request import urlopen
import sys
from bs4 import BeautifulSoup
import bs4
import json
import ast


class ClassScraper:
    def __init__(self):
        self.module_urls = []
        self.class_urls = []
        self.classes = []
        self.library = ""

    def get_classes(self):
        self.scrape_class_urls()
        self.scrape_classes()
        self.create_json()

    def scrape_parameters(self, elements):
        parameters = {}
        for e in elements:
            split = e.text.split("=", 1)
            param = split[0].split(":")[0]
            if param == "*":
                continue
            if param[0].isnumeric():
                last_key = list(parameters)[-1]
                parameters[last_key] += ", {0}".format(param)
            elif len(split) > 1:
                default_value = split[1].strip()
                parameters[param] = default_value
            else:
                parameters[param] = None
        return parameters

    def create_json(self):
        with open("output/scraped_classes/{0}_default_values.json".format(self.library), "w") as outfile:
            json.dump(self.classes, outfile, indent=4)


class SklearnScraper(ClassScraper):
    def __init__(self):
        ClassScraper.__init__(self)
        self.library = "sklearn"

    def scrape_class_urls(self):
        link = "https://scikit-learn.org/stable/modules/classes.html#"
        html = urlopen(link)
        soup = BeautifulSoup(html, "html.parser")

        table_rows = soup.body.findAll("tr")
        for row in table_rows:
            #text = row.find("a").text
            #text = text[text.rfind(".") + 1:]
            #if text[0].isupper():
            url = row.find("a").attrs["href"]
            #print("url: ", url)
            self.class_urls.append(url)

    def scrape_classes(self):
        full_names = []
        for url in self.class_urls:
            try:
                link = "https://scikit-learn.org/stable/modules/" + url
                html = urlopen(link)
                soup = BeautifulSoup(html, "html.parser")

                data_table = soup.find("dt", {"class": "sig sig-object py"})
                full_class_name = data_table.attrs["id"]
                class_ = full_class_name[full_class_name.rfind(".") + 1:]
                elements = data_table.findAll("em", {"class": "sig-param"})

                parameters = self.scrape_parameters(elements)

                # params = list(key for key in parameters.keys() if key != "*")
                if full_class_name not in full_names:
                    self.classes.append({"full_name": full_class_name, "name": class_, "params": parameters})
                    full_names.append(full_class_name)

                #self.classes[full_class_name] = {"short name": class_, "parameters": parameters}

            except:
                print("Could not crawl: ", url)


class PyTorchScraper(ClassScraper):
    def __init__(self):
        ClassScraper.__init__(self)
        self.library = "torch"
        self.desc_classes = []
        self.desc_functions = []

    def get_classes(self):
        self.scrape_module_urls()
        self.scrape_class_urls()
        self.scrape_desc_elements()
        self.scrape_classes()
        self.create_json()

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
                            #title = a.attrs["title"]
                            #title = title[title.rfind(".") + 1:]
                            #if title[0].isupper():
                            url = row.find("a").attrs["href"]
                            self.class_urls.append(url)

                    dl = soup.findAll("dl", {"class": "py class"})
                    self.desc_classes.extend(dl)
            except:
                continue

        autograd_urls = ["generated/torch.autograd.inference_mode.html",
                         "generated/torch.autograd.set_grad_enabled.html",
                         "generated/torch.autograd.enable_grad.html",
                         "generated/torch.autograd.no_grad.html",
                         "generated/torch.autograd.forward_ad.dual_level.html"]
        for url in autograd_urls:
            self.class_urls.append(url)

    def scrape_desc_elements(self):
        for url in self.class_urls:
            link = "https://pytorch.org/docs/stable/" + url
            html = urlopen(link)
            #print("Link: ", link)
            soup = BeautifulSoup(html, "html.parser")
            dl_classes = soup.findAll("dl", {"class": "py class"})
            if len(dl_classes) > 0:
                for element in dl_classes:
                    self.desc_classes.append(element)
            dl_functions = soup.findAll("dl", {"class": "py function"})
            if len(dl_functions) > 0:
                for element in dl_functions:
                    self.desc_functions.append(element)


    def scrape_classes(self):
        full_names = []
        for desc_class in self.desc_classes:
            data_table = desc_class.find("dt")
            if "id" in data_table.attrs:
                class_path = data_table.attrs["id"]
                class_ = class_path[class_path.rfind(".") + 1:]
                elements = data_table.findAll("em", {"class": "sig-param"})
                parameters = self.scrape_parameters(elements)
                if class_path == "torch.torch.device":
                    parameters = {"type": None, "index": None}

                #params = list(key for key in parameters.keys() if key != "*")

                if class_path not in full_names:
                    self.classes.append({"full_name": class_path, "name": class_, "params": parameters})
                    full_names.append(class_path)

                #self.classes[class_path] = {"name": class_, "parameters": parameters}
            

        for desc_function in self.desc_functions:
            data_table = desc_function.find("dt")
            if "id" in data_table.attrs:
                class_path = data_table.attrs["id"]
                class_ = class_path[class_path.rfind(".") + 1:]
                elements = data_table.findAll("em", {"class": "sig-param"})
                parameters = self.scrape_parameters(elements)
                if class_path == "torch.torch.device":
                    parameters = {"type": None, "index": None}

                #params = list(key for key in parameters.keys() if key != "*")
                if class_path not in full_names:
                    self.classes.append({"full_name": class_path, "name": class_, "params": parameters})
                    full_names.append(class_path)

                #self.classes[class_path] = {"short name": class_, "parameters": parameters}


class MLflowScraper(ClassScraper):
    def __init__(self):
        ClassScraper.__init__(self)
        self.library = "mlflow"

    def get_classes(self):
        self.scrape_module_urls()
        self.scrape_classes()
        self.create_json()

    def scrape_module_urls(self):
        link = "https://mlflow.org/docs/latest/python_api/index.html"
        html = urlopen(link)
        soup = BeautifulSoup(html, "html.parser")

        div = soup.find("div", {"class": "section"})
        li = div.findAll("li", {"class": "toctree-l1"})

        for element in li:
            url = element.find("a").attrs["href"]
            self.module_urls.append(url)

    def scrape_classes(self):
        for url in self.module_urls:
            link = "https://mlflow.org/docs/latest/python_api/" + url
            html = urlopen(link)
            soup = BeautifulSoup(html, "html.parser")

            dl = soup.findAll("dl", {"class": "py class"})
            for element in dl:
                dt = element.find("dt")
                full_class_name = dt.attrs["id"]
                class_ = full_class_name[full_class_name.rfind(".") + 1:]
                em = dt.findAll("em", {"class": "sig-param"})
                parameters = self.scrape_parameters(em)
                self.classes[full_class_name] = {"short name": class_, "parameters": parameters}

        self.classes["mlflow.models.Model"]["parameters"] = {"artifact_path": "None", "run_id": "None",
                                                             "utc_time_created": "None",
                                                             "flavors": "None", "signature": "None",
                                                             "saved_input_example_info": "None",
                                                             "model_uuid": "<function Model.<lambda>>",
                                                             "**kwargs": "None"}


class TensorFlowScraper(ClassScraper):
    def __init__(self):
        ClassScraper.__init__(self)
        self.library = "tensorflow"

    def get_classes(self):
        self.scrape_class_urls()
        self.scrape_classes()
        self.create_json()

    def scrape_class_urls(self):
        link = "https://www.tensorflow.org/api_docs/python/tf"
        self.module_urls.append(link)
        for url in self.module_urls:
            html = urlopen(url)
            soup = BeautifulSoup(html, "html.parser")

            self.scrape_urls(soup, "modules")
            self.scrape_urls(soup, "classes")
            self.scrape_urls(soup, "functions")

    def scrape_urls(self, soup, type):
        element = soup.find("h2", {"id": type})
        if element is None:
            element = soup.find("h2", {"id": "{0}_2".format(type)})
            if element is None:
                return
        while True:
            element = element.nextSibling
            if element is None:
                break
            if isinstance(element, bs4.Tag):
                if element.name == "h2":
                    break
                try:
                    url = element.find("a").attrs["href"]
                except:
                    continue
                if type == "modules":
                    if url not in self.module_urls:
                        self.module_urls.append(url)
                elif type == "classes":
                    if url not in self.class_urls:
                        self.class_urls.append(url)
                elif type == "functions":
                    if url not in self.class_urls:
                        self.class_urls.append(url)

    def scrape_classes(self):
        full_names = []

        for url in self.class_urls:
            #print("url: ", url)
            html = urlopen(url)
            soup = BeautifulSoup(html, "html.parser")

            full_class_name = soup.find("h1", {"class": "devsite-page-title"}).text
            class_ = full_class_name[full_class_name.rfind(".") + 1:]
            pre = soup.find("pre")
            parameters = self.scrape_parameters(pre, full_class_name)
            full_class_name = full_class_name.replace(full_class_name[:2], "tensorflow")

            #params = list(parameters.keys())

            if full_class_name not in full_names:
                self.classes.append({"full_name": full_class_name, "name": class_, "params": parameters})
                full_names.append(full_class_name)

            #self.classes[full_class_name] = {"short name": class_, "parameters": parameters}

    def scrape_parameters(self, pre, full_class_name):
        parameters = {}
        if pre is not None:
            if pre.text.startswith("\n{0}".format(full_class_name)):
                try:
                    ast_obj = ast.parse(pre.text).body[0].value
                    for arg in ast_obj.args:
                        parameters[ast.unparse(arg)] = None
                    for keyword in ast_obj.keywords:
                        if ast.unparse(keyword.value) != "kwargs":
                            parameters[keyword.arg] = ast.unparse(keyword.value)
                        else:
                            parameters["**kwargs"] = None
                except:
                    if full_class_name == "tf.lite.experimental.QuantizationDebugOptions":
                        parameters = {"layer_debug_metrics": None, "model_debug_metrics": None,
                                      "layer_direct_compare_metrics": None, "denylisted_ops": None,
                                      "denylisted_nodes": None, "fully_quantize": False}
                    elif full_class_name == "tf.lite.experimental.QuantizationDebugger":
                        parameters = {"quant_debug_model_path": None, "quant_debug_model_content": None,
                                      "float_model_path": None, "float_model_content": None,
                                      "debug_dataset": None, "debug_options": None, "converter": None}

        if full_class_name == "tf.compat.v1.ConfigProto":
            parameters = {"allow_soft_placement": None, "cluster_def": None, "device_count": None,
                          "device_filters": None, "experimental": None, "gpu_options": None,
                          "graph_options": None, "inter_op_parallelism_threads": None,
                          "intra_op_parallelism_threads": None, "isolate_session_state": None,
                          "log_device_placement": None, "operation_timeout_in_ms": None, "placement_period": None,
                          "rpc_options": None, "session_inter_op_thread_pool": None,
                          "share_cluster_devices_in_session": None, "use_per_session_threads": None}
        return parameters


lib_dict = {"sklearn": SklearnScraper,
            "Sklearn": SklearnScraper,
            "scikit-learn": SklearnScraper,
            "Scikit-learn": SklearnScraper,
            "Scikitlearn": SklearnScraper,
            "skl": SklearnScraper,
            "tensorflow": TensorFlowScraper,
            "TensorFlow": TensorFlowScraper,
            "Tensorflow": TensorFlowScraper,
            "tf": TensorFlowScraper,
            "mlflow": MLflowScraper,
            "MLFLow": MLflowScraper,
            "MlFlow": MLflowScraper,
            "MLflow": MLflowScraper,
            "pytorch": PyTorchScraper,
            "PyTorch": PyTorchScraper,
            "Pytorch": PyTorchScraper,
            "Torch": PyTorchScraper,
            "torch": PyTorchScraper}


def main():
    library = sys.argv[1]
    scraper = lib_dict[library]
    scraper().get_classes()


if __name__ == "__main__":
    main()
