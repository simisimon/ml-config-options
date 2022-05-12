import ast
import json
import git
import sys
from os import walk, path
from obj_selector import PyTorchObjects, SklearnObjects, MLflowObjects, TensorFlowObjects
from node import NodeObject
from dataflow import DataFlowAnalysis


class NodeObjects:
    def __init__(self, repo):
        self.library = ""
        self.classes = {}
        self.repo = repo
        self.py_files = []
        self.class_objects_from_library = []
        self.node_objects = []

    def get_nodes(self):
        self.get_py_files()
        self.get_class_objects()
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

    def get_parameters(self):
        for obj in self.class_objects_from_library:
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
                                   "object": ast.parse(obj_code).body[0],
                                   "parameter variables": obj["parameter variables"]}
                    except:
                        new_obj = {"file": obj["file"], "class": obj["class"], "class alias": obj["class alias"],
                                   "object": ast.parse("{0}[]".format(obj_code)).body[0],
                                   "parameter variables": obj["parameter variables"]}
                    node_obj = NodeObject(new_obj).get_objects(new_obj["object"])
                    node_obj["line_no"] = obj["object"].lineno
                    self.node_objects.append(node_obj)
                    obj_code = ast.unparse(obj["object"])
            else:
                node_obj = NodeObject(obj).get_objects(obj["object"])
                self.node_objects.append(node_obj)

    def merge_parameter(self):
        for obj in self.node_objects:
            class_parameters = self.classes[obj["class"]]
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
        for obj in self.node_objects:
            for variable in obj["parameter variables"]:
                obj["parameter variables"][variable] = DataFlowAnalysis(obj, variable).get_parameter_value()

    def convert_into_node_structure(self):
        json_nodes = []
        for obj in self.node_objects:
            dict_obj = {"file": obj["file"].split("/", 1)[-1],
                        "class": obj["class"],
                        "line_no": obj["line_no"],
                        "variable": None,
                        "parameter": obj["parameter"],
                        "parameter_values": obj["parameter variables"]}
            json_nodes.append(dict_obj)

        with open("nodes/{0}_nodes.txt".format(self.library), 'w') as outfile:
            json.dump(json_nodes, outfile, indent=4)


class SklearnNodes(NodeObjects):
    def __init__(self, repo):
        NodeObjects.__init__(self, repo)
        self.library = "sklearn"
        self.classes = SklearnObjects("").read_json()

    def get_class_objects(self):
        for file in self.py_files:
            self.class_objects_from_library.extend(SklearnObjects(file).get_objects())


class PyTorchNodes(NodeObjects):
    def __init__(self, repo):
        NodeObjects.__init__(self, repo)
        self.library = "torch"
        self.classes = PyTorchObjects("").read_json()

    def get_class_objects(self):
        for file in self.py_files:
            self.class_objects_from_library.extend(PyTorchObjects(file).get_objects())


class MLflowNodes(NodeObjects):
    def __init__(self, repo):
        NodeObjects.__init__(self, repo)
        self.library = "mlflow"
        self.classes = MLflowObjects("").read_json()

    def get_class_objects(self):
        for file in self.py_files:
            self.class_objects_from_library.extend(MLflowObjects(file).get_objects())


class TensorFlowNodes(NodeObjects):
    def __init__(self, repo):
        NodeObjects.__init__(self, repo)
        self.library = "tensorflow"
        self.classes = TensorFlowObjects("").read_json()

    def get_class_objects(self):
        for file in self.py_files:
            self.class_objects_from_library.extend(TensorFlowObjects(file).get_objects())


lib_dict = {"sklearn": SklearnNodes,
            "Sklearn": SklearnNodes,
            "Scikit-learn": SklearnNodes,
            "scikit-learn": SklearnNodes,
            "Scikitlearn": SklearnNodes,
            "skl": SklearnNodes,
            "tensorflow": TensorFlowNodes,
            "TensorFlow": TensorFlowNodes,
            "Tensorflow": TensorFlowNodes,
            "tf": TensorFlowNodes,
            "mlflow": MLflowNodes,
            "MLFLow": MLflowNodes,
            "MlFlow": MLflowNodes,
            "MLflow": MLflowNodes,
            "pytorch": PyTorchNodes,
            "PyTorch": PyTorchNodes,
            "Pytorch": PyTorchNodes,
            "Torch": PyTorchNodes,
            "pytorch": PyTorchNodes,
            "torch": PyTorchNodes}


def main():
    repo_path = sys.argv[1]
    library = sys.argv[2]

    repo_name = repo_path.split('/')[-1]
    is_directory = path.isdir(repo_name)
    if not is_directory:
        git.Repo.clone_from('https://github.com/mj-support/coop.git', repo_name)

    nodes = lib_dict[library]
    nodes(repo_name).get_nodes()


if __name__ == "__main__":
    main()
