import ast
from pprint import pprint
import json


class CodeObjects:
    def __init__(self, project):
        self.library = ""
        self.classes = {}
        self.project = project
        self.first_level_objects = []
        self.import_objects = []
        self.function_objects = []  # scope of parameter variables
        self.objects_from_library = []
        self.library_class_objects = []

    def get_objects(self):
        self.read_json()
        self.get_first_level_objects()
        self.get_import_objects()
        self.get_objects_from_library()
        self.get_objects_containing_classes()
        self.get_param_variables()
        return self.library_class_objects

    def read_json(self):
        with open("{0}.txt".format(self.library)) as json_file:
            self.classes = json.load(json_file)
        return self.classes

    def get_first_level_objects(self):
        with open(self.project, "r") as source:
            tree = ast.parse(source.read())
        self.first_level_objects = tree.body

    def get_import_objects(self):
        for obj in self.first_level_objects:
            nodes = list(ast.walk(obj))
            for node in nodes:
                if type(node) == ast.Import:
                    for package in node.names:
                        if self.library in package.name:
                            if package.asname is not None:
                                self.import_objects.append(package.asname)
                            else:
                                self.import_objects.append(package.name)
                elif type(node) == ast.ImportFrom:
                    for package in node.names:
                        if node.module is not None:
                            if self.library in node.module:
                                if package.asname is not None:
                                    self.import_objects.append(package.asname)
                                else:
                                    self.import_objects.append(package.name)

    def get_objects_from_library(self):
        for obj in self.first_level_objects:
            self.get_object(obj)

    def get_object(self, obj):
        if hasattr(obj, 'body'):
            if type(obj) == ast.FunctionDef or type(obj) == ast.AsyncFunctionDef:
                assignments = []
                for body_obj in obj.body:
                    if type(body_obj) == ast.Assign:
                        assignments.append(body_obj)
                func = {"line no": obj.lineno, "line no end": obj.end_lineno, "assignments": assignments}
                self.function_objects.append(func)
            for body_obj in obj.body:  # func, async func, class, with, async with, except handler
                self.get_object(body_obj)
            obj.body = []
            if hasattr(obj, 'orelse'):  # control-flow-objects: for, if, async for, while
                for orelse_obj in obj.orelse:
                    self.get_object(orelse_obj)
                obj.orelse = []
                if hasattr(obj, 'handlers'):  # exception-objects: try
                    for handler_obj in obj.handlers:
                        self.get_object(handler_obj)
                    for finalbody_obj in obj.finalbody:
                        self.get_object(finalbody_obj)
                    obj.handlers = []
                    obj.finalbody = []
        if type(obj) != ast.Import and type(obj) != ast.ImportFrom:  # import-objects: import, import from
            dump_ast_obj = ast.dump(obj)
            for import_obj in self.import_objects:
                import_string = "'{0}'".format(import_obj)
                if import_string in dump_ast_obj:
                    self.objects_from_library.append(obj)
                    break

    def get_objects_containing_classes(self):
        for obj in self.objects_from_library:
            dump_ast_obj = ast.dump(obj)
            for class_ in self.classes:
                class_edit = "'{0}'".format(class_)
                if class_edit in dump_ast_obj:
                    class_string = "{0}(".format(class_)
                    obj_code = ast.unparse(obj)
                    indices = [i for i in range(len(obj_code)) if obj_code.startswith(class_string, i)]
                    for index in indices:
                        if not(obj_code[index - 1].isalnum()) or index == 0:
                            if class_string in ast.unparse(obj):
                                class_object = {"class": class_, "object": obj, "parameter variables": None}
                                self.library_class_objects.append(class_object)

    def get_param_variables(self):
        global_assignments = []
        for obj in self.first_level_objects:
            if type(obj) == ast.Assign:
                global_assignments.append(obj)
        self.function_objects = sorted(self.function_objects, key=lambda d: d["line no end"])
        for obj in self.library_class_objects:
            for func in self.function_objects:
                if func["line no"] <= obj["object"].lineno:
                    if obj["object"].lineno <= func["line no end"]:
                        assigns_before_obj = []
                        for assign in func["assignments"]:
                            if assign.lineno < obj["object"].lineno:
                                assigns_before_obj.append(assign)
                            else:
                                break
                        obj["parameter variables"] = assigns_before_obj
                        break
                else:
                    global_assigns_before_obj = []
                    for assign in global_assignments:
                        if assign.lineno < obj["object"].lineno:
                            global_assigns_before_obj.append(assign)
                        else:
                            break
                    obj["parameter variables"] = global_assigns_before_obj
                    break


class SklearnObjects(CodeObjects):
    def __init__(self, project):
        CodeObjects.__init__(self, project)
        self.library = "sklearn"


class TorchObjects(CodeObjects):
    def __init__(self, project):
        CodeObjects.__init__(self, project)
        self.library = "torch"


def main():
    project = "test_projects/another_test_project.py"
    # project = "test_projects/torch_project.py"
    ast_objects = SklearnObjects(project).get_objects()
    # ast_objects = TorchObjects(project).get_objects()

    pprint(ast_objects, width=75)


if __name__ == "__main__":
    main()
