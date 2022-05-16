import ast
import json
import git
import sys
from os import walk, path
from ast_classes import ASTClasses
from ast_parameters import ASTParameters
from dataflow import DataFlowAnalysis


class ConfigOptions:
    def __init__(self, repo):
        self.library = ""
        self.scraped_classes = ""
        self.repo = repo
        self.py_files = []
        self.ast_classes = []
        self.config_objects = []

    def get_config_options(self):
        self.scraped_classes = ASTClasses(None, self.library).read_json()
        self.get_py_files()
        self.get_ast_classes()
        self.get_parameters()
        self.merge_parameter()
        self.get_parameter_values()
        self.convert_into_node_structure()

    def get_py_files(self):
        for root, dirs, files in walk(self.repo):
            for filename in files:
                if filename.endswith(".py"):
                    file = path.join(root, filename)
                    self.py_files.append(file)

    def get_ast_classes(self):
        for file in self.py_files:
            self.ast_classes.extend(ASTClasses(file, self.library).get_classes())

    def get_parameters(self):
        for obj in self.ast_classes:
            obj_code = ast.unparse(obj["object"])
            class_string = "{0}(".format(obj["class alias"])
            indices = [i for i in range(len(obj_code)) if obj_code.startswith(class_string, i)]
            for index in indices.copy():
                if index != 0 and (obj_code[index - 1].isalnum() or obj_code[index - 1] == "."):
                    indices.remove(index)
            indices.reverse()

            """check if same class occurs multiple times in obj"""
            if len(indices) > 1:
                for i in range(len(indices)):
                    for j in range(len(indices)):
                        if i != j:
                            start = indices[j]
                            stop = start + len(obj["class alias"])
                            obj_code = "".join((obj_code[:start], "temp", obj_code[stop:]))
                    try:
                        new_obj = {"file": obj["file"], "class": obj["class"], "class alias": obj["class alias"],
                                   "object": ast.parse(obj_code).body[0]}
                    except:
                        new_obj = {"file": obj["file"], "class": obj["class"], "class alias": obj["class alias"],
                                   "object": ast.parse("{0}[]".format(obj_code)).body[0]}
                    config_object = ASTParameters(new_obj).get_parameters(new_obj["object"])
                    self.config_objects.append(config_object)
                    obj_code = ast.unparse(obj["object"])
            else:
                config_object = ASTParameters(obj).get_parameters(obj["object"])
                self.config_objects.append(config_object)

    def merge_parameter(self):
        for obj in self.config_objects:
            class_parameters = self.scraped_classes[obj["class"]]
            project_parameters = obj["parameter"]
            parameter_dict = {}
            asterisks = False
            for prj_param in project_parameters:
                if prj_param[0] is None and not asterisks:
                    for cls_param in list(class_parameters["parameters"]):
                        if cls_param[0] == "*" and len(cls_param) > 1:
                            if cls_param[1].isalpha():  # handling of *args
                                if cls_param not in parameter_dict:
                                    parameter_dict[cls_param] = []
                                parameter_dict[cls_param].append(prj_param[1])
                            break
                        elif len(parameter_dict) <= list(class_parameters["parameters"]).index(cls_param):
                            if cls_param == "*":
                                asterisks = True
                                break
                            else:
                                parameter_dict[cls_param] = prj_param[1]
                                break
                else:
                    for cls_param in list(class_parameters["parameters"]):
                        if cls_param == prj_param[0]:
                            parameter_dict[cls_param] = prj_param[1]
                            break
                        elif cls_param[:2] == "**":  # handling of **kwargs
                            parameter_dict[prj_param[0]] = prj_param[1]

            obj["parameter"] = parameter_dict

    def get_parameter_values(self):
        for obj in self.config_objects:
            for variable in obj["parameter variables"]:
                obj["parameter variables"][variable] = DataFlowAnalysis(obj, variable).get_parameter_value()

    def convert_into_node_structure(self):
        json_nodes = []
        for obj in self.config_objects:
            dict_obj = {"file": obj["file"].split("/", 1)[-1],
                        "class": obj["class"],
                        "line_no": obj["object"].lineno,
                        "variable": None,
                        "parameter": obj["parameter"],
                        "parameter_values": obj["parameter variables"]}
            json_nodes.append(dict_obj)

        with open("nodes/{0}_nodes.txt".format(self.library), 'w') as outfile:
            json.dump(json_nodes, outfile, indent=4)


class SklearnOptions(ConfigOptions):
    def __init__(self, repo):
        ConfigOptions.__init__(self, repo)
        self.library = "sklearn"


class PyTorchOptions(ConfigOptions):
    def __init__(self, repo):
        ConfigOptions.__init__(self, repo)
        self.library = "torch"


class MLflowOptions(ConfigOptions):
    def __init__(self, repo):
        ConfigOptions.__init__(self, repo)
        self.library = "mlflow"


class TensorFlowOptions(ConfigOptions):
    def __init__(self, repo):
        ConfigOptions.__init__(self, repo)
        self.library = "tensorflow"


def clone_repo(repo_path):
    repo_name = repo_path.split('/')[-1]
    is_directory = path.isdir(repo_name)
    if not is_directory:
        git.Repo.clone_from(repo_path, repo_name)
    return repo_name


lib_dict = {"sklearn": SklearnOptions,
            "Sklearn": SklearnOptions,
            "Scikit-learn": SklearnOptions,
            "scikit-learn": SklearnOptions,
            "Scikitlearn": SklearnOptions,
            "skl": SklearnOptions,
            "tensorflow": TensorFlowOptions,
            "TensorFlow": TensorFlowOptions,
            "Tensorflow": TensorFlowOptions,
            "tf": TensorFlowOptions,
            "Tf": TensorFlowOptions,
            "mlflow": MLflowOptions,
            "MLFLow": MLflowOptions,
            "MlFlow": MLflowOptions,
            "MLflow": MLflowOptions,
            "pytorch": PyTorchOptions,
            "PyTorch": PyTorchOptions,
            "Pytorch": PyTorchOptions,
            "Torch": PyTorchOptions,
            "pytorch": PyTorchOptions,
            "torch": PyTorchOptions}


def main():
    repo_path = 'https://github.com/mj-support/coop'  # sys.argv[1]
    library = 'sklearn'  # sys.argv[2]

    repo_name = clone_repo(repo_path)
    try:
        library_cap = library[0].upper() + library[1:]
        eval("{0}Options('{1}').get_config_options()".format(library_cap, repo_name))
    except:
        options = lib_dict[library]
        options(repo_name).get_config_options()


if __name__ == "__main__":
    main()
