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
        self.get_variable_parameter_values()
        self.create_json()

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
        for ast_class_dict in self.ast_classes:
            scraped_parameters = self.scraped_classes[ast_class_dict["class"]]

            obj_code = ast.unparse(ast_class_dict["object"])
            class_string = "{0}(".format(ast_class_dict["class alias"])
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
                            stop = start + len(ast_class_dict["class alias"])
                            obj_code = "".join((obj_code[:start], "temp", obj_code[stop:]))
                    try:
                        new_obj = {"file": ast_class_dict["file"], "class": ast_class_dict["class"],
                                   "class alias": ast_class_dict["class alias"], "object": ast.parse(obj_code).body[0],
                                   "scraped parameters": scraped_parameters}
                    except:
                        new_obj = {"file": ast_class_dict["file"], "class": ast_class_dict["class"],
                                   "class alias": ast_class_dict["class alias"], "object": ast.parse("{0}[]".format(obj_code)).body[0],
                                   "scraped parameters": scraped_parameters}
                    config_object = ASTParameters(new_obj).get_parameters(new_obj["object"])
                    self.config_objects.append(config_object)
                    obj_code = ast.unparse(ast_class_dict["object"])
            else:
                ast_class_dict["scraped parameters"] = scraped_parameters
                config_object = ASTParameters(ast_class_dict).get_parameters(ast_class_dict["object"])
                self.config_objects.append(config_object)

    def get_variable_parameter_values(self):
        for obj in self.config_objects:
            for variable in obj["variable parameters"]:
                obj["variable parameters"][variable] = DataFlowAnalysis(obj, variable).get_parameter_value()

    def create_json(self):
        for obj in self.config_objects:
            obj["file"] = obj["file"][obj["file"].find('/') + 1:]
            obj["line no"] = obj["object"].lineno
            obj.pop("object")

        with open("output/config_options/{0}_options.txt".format(self.library), 'w') as outfile:
            json.dump(self.config_objects, outfile, indent=4)


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


def clone_repo(repo_link):
    repo_name = repo_link.split('/')[-1]
    repo_dir = 'input_repo/{0}'.format(repo_name)
    is_directory = path.isdir(repo_dir)
    if not is_directory:
        git.Repo.clone_from(repo_link, repo_dir)
    return repo_dir


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
    repo_link = sys.argv[1] #'https://github.com/mj-support/coop'  # sys.argv[1]
    library = sys.argv[2]  #'scikit-learn'  # sys.argv[2]

    repo_dir = clone_repo(repo_link)
    try:
        library_cap = library[0].upper() + library[1:]
        eval("{0}Options('{1}').get_config_options()".format(library_cap, repo_dir))
    except:
        options = lib_dict[library]
        options(repo_dir).get_config_options()


if __name__ == "__main__":
    main()
