import ast
from pprint import pprint
import json


class CodeObjects:
    def __init__(self, project):
        self.library = ""
        self.classes = {}
        self.project = project
        self.first_level_objects = []
        self.import_classes = {}
        self.import_other_objects = {}
        self.function_objects = []  # scope of parameter variables
        self.objects_from_library = []
        self.class_objects_from_library = []

    def get_objects(self):
        self.read_json()
        self.get_first_level_objects()
        self.get_import_objects()
        self.get_objects_from_library()
        self.get_objects_containing_classes()
        self.get_param_variables()
        return self.class_objects_from_library

    def read_json(self):
        with open("{0}.txt".format(self.library)) as json_file:
            self.classes = json.load(json_file)
        return self.classes

    def get_first_level_objects(self):
        with open(self.project, "r") as source:
            tree = ast.parse(source.read())
        self.first_level_objects = tree.body

    def get_import_objects(self):
        all_import_objects = {}
        for obj in self.first_level_objects:
            nodes = list(ast.walk(obj))
            for node in nodes:
                if type(node) == ast.Import:
                    for package in node.names:
                        if self.library in package.name:
                            if package.asname is not None:
                                all_import_objects[package.asname] = {"path": package.name}
                            else:
                                all_import_objects[package.name] = {"path": package.name}

                elif type(node) == ast.ImportFrom:
                    for package in node.names:
                        module = node.module
                        if module is not None:
                            if self.library in module:
                                if package.asname is not None:
                                    all_import_objects[package.asname] = {"path": "{0}.{1}".format(module, package.name)}
                                else:
                                    all_import_objects[package.name] = {"path": "{0}.{1}".format(module, package.name)}

        for import_obj in all_import_objects:
            is_class = False
            for class_ in self.classes:
                if class_ == all_import_objects[import_obj]["path"]:
                    is_class = True
                    break

            splitted_import_obj = import_obj.split(".")
            splitted_import_obj[0] = "id='{0}'".format(splitted_import_obj[0]) #fehler wegen ID!!
            for sio in enumerate(splitted_import_obj):
                if sio[0] > 0:
                    splitted_import_obj[sio[0]] = "attr='{0}'".format(sio[1])
            all_import_objects[import_obj].update({"ast style": splitted_import_obj})

            if is_class:
                self.import_classes[import_obj] = all_import_objects[import_obj]
            else:
                self.import_other_objects[import_obj] = all_import_objects[import_obj]

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
            import_values = dict(self.import_classes)
            import_values.update(self.import_other_objects)
            for import_obj in import_values.values():
                import_obj_findings = [i for i in import_obj["ast style"] if i in dump_ast_obj]
                if import_obj["ast style"] == import_obj_findings:
                    self.objects_from_library.append(obj)
                    break

    def get_objects_containing_classes(self):
        for import_name, import_obj_values in self.import_other_objects.items():     # to detect classes that are not declared as the common path
            if import_obj_values['path'] == self.library:
                library_name_import = True
                library_alias = import_name

        for obj in self.objects_from_library:
            dump_ast_obj = ast.dump(obj)
            obj_code = ast.unparse(obj)
            for import_class_name, import_class_values in self.import_classes.items():
                class_occurence = 0
                for import_value in import_class_values["ast style"]:
                    class_string = "{0}(".format(import_class_name)
                    if class_string in obj_code:
                        indices = [i for i in range(len(obj_code)) if obj_code.startswith(class_string, i)]
                        for index in indices.copy():
                            if index == 0 or not(obj_code[index - 1].isalnum() or obj_code[index - 1] == "."):
                                class_occurence += 1
                                dump_ast_obj = dump_ast_obj.replace(import_value, "id='temp_class'", 1)
                                obj_code = obj_code.replace(class_string, "temp_class(", 1)
                            else:
                                indices.remove(index)
                        if class_occurence > 0:
                            obj_code = obj_code.replace("temp_class(", class_string)
                            lineno = obj.lineno
                            end_lineno = obj.end_lineno
                            try:
                                obj = ast.parse(obj_code).body[0]
                            except:
                                obj = ast.parse("{0}[]".format(obj_code)).body[0]
                            obj.lineno = lineno
                            obj.end_lineno = end_lineno
                            lib_class_obj = {"class": import_class_values["path"], "class alias": class_string[:-1], "object": obj, "parameter variables": None}
                            self.class_objects_from_library.append(lib_class_obj)

            for import_name, import_obj_values in self.import_other_objects.items():
                import_obj_findings = [i for i in import_obj_values["ast style"] if i in dump_ast_obj]
                if import_obj_values["ast style"] == import_obj_findings:
                    for class_ in self.classes:
                        class_occurence = 0
                        if import_obj_values["path"] in class_:
                            class_string = "{0}(".format(class_.replace(import_obj_values["path"], import_name))
                            if class_string in obj_code:
                                indices = [i for i in range(len(obj_code)) if obj_code.startswith(class_string, i)]
                                for index in indices:
                                    if index == 0 or not (obj_code[index - 1].isalnum() or obj_code[index - 1] == "."):
                                        class_occurence += 1
                                        dump_ast_obj = dump_ast_obj.replace(import_value, "id='temp_class'", 1)
                                        obj_code = obj_code.replace(class_string, "temp_class(", 1)
                                if class_occurence > 0:
                                    if library_name_import:
                                        if class_.split(".")[-1] in obj_code and library_alias in obj_code:
                                            split_code = obj_code.split(library_alias)
                                            split_class = class_.split(".")[1:]
                                            for split in split_class:
                                                split_class[split_class.index(split)] = ".{0}.".format(split)
                                            split_class[-1] = "{0}(".format(split_class[-1][:-1])

                                            for split in split_code:
                                                split = split.split(" ")[0]
                                                if split.startswith("."):
                                                    findings = [i for i in split_class if i in split]
                                                    if split_class == findings:
                                                        start = obj_code.index(library_alias + split)
                                                        stop = obj_code.index("(", start)
                                                        obj_code = obj_code[:start] + "temp_class(" + obj_code[stop + 1:]

                                            obj_code = obj_code.replace("temp_class(", class_string)
                                            lineno = obj.lineno
                                            end_lineno = obj.end_lineno
                                            try:
                                                obj = ast.parse(obj_code).body[0]
                                            except:
                                                obj = ast.parse("{0}[]".format(obj_code)).body[0]
                                            obj.lineno = lineno
                                            obj.end_lineno = end_lineno

                                    lib_class_obj = {"class": class_, "class alias": class_string[:-1], "object": obj,
                                                     "parameter variables": None}
                                    self.class_objects_from_library.append(lib_class_obj)

    def get_param_variables(self):
        global_assignments = []
        for obj in self.first_level_objects:
            if type(obj) == ast.Assign:
                global_assignments.append(obj)
        self.function_objects = sorted(self.function_objects, key=lambda d: d["line no end"])
        for obj in self.class_objects_from_library:
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
    #project = "test_projects/another_test_project.py"
    project = "test_projects/torch_project.py"
    #ast_objects = SklearnObjects(project).get_objects()
    ast_objects = TorchObjects(project).get_objects()

    pprint(ast_objects, width=75)


if __name__ == "__main__":
    main()
