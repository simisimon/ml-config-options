import ast
from pprint import pprint
import json


class CodeObjects:
    def __init__(self, ml_lib, project):
        self.ml_lib = ml_lib
        self.project = project
        self.functions = []
        self.preselected_ast_objects = []
        self.final_ast_objects = []


    def get_objects(self):
        classes = self.read_json()
        parent_ast_objects = self.get_parent_ast_objects()
        import_values = self.get_import_val(parent_ast_objects)
        self.preselect_ast_objects(parent_ast_objects, import_values)
        self.get_ast_objects_containing_classes(classes)
        self.get_param_variables(parent_ast_objects)
        return self.final_ast_objects

    def read_json(self):
        with open(self.ml_lib + '.txt') as json_file:
            classes = json.load(json_file)
        return classes

    def get_parent_ast_objects(self):
        with open(self.project, "r") as source:
            tree = ast.parse(source.read())
        obj = tree.body
        return obj

    def get_import_val(self, parent_ast_objects):
        import_obj = []
        for o in parent_ast_objects:
            desc_nodes = list(ast.walk(o))
            for d in desc_nodes:
                if type(d) == ast.Import:
                    for n in d.names:
                        if self.ml_lib in n.name:
                            if n.asname != None:
                                import_obj.append(n.asname)
                            else:
                                import_obj.append(n.name)
                elif type(d) == ast.ImportFrom:
                    for n in d.names:
                        if d.module != None:
                            if self.ml_lib in d.module:
                                if n.asname != None:
                                    import_obj.append(n.asname)
                                else:
                                    import_obj.append(n.name)
        return import_obj

    def preselect_ast_objects(self, parent_obj, import_val):
        for p in parent_obj:
            self.get_obj(p, import_val)

    def get_obj(self, obj, import_val):
        if hasattr(obj, 'body'):
            if type(obj) == ast.FunctionDef or type(obj) == ast.AsyncFunctionDef:
                assignments = []
                for body_obj in obj.body:
                    if type(body_obj) == ast.Assign:
                        assignments.append(body_obj)
                func = {"line no": obj.lineno, "line no end": obj.end_lineno, "assignments": assignments}
                self.functions.append(func)
            for b in obj.body:  # func, async func, class, with, async with, except handler
                self.get_obj(b, import_val)
            obj.body = []
            if hasattr(obj, 'orelse'):  # control-flow-objects: for, if, async for, while
                for o in obj.orelse:
                    self.get_obj(o, import_val)
                obj.orelse = []
                if hasattr(obj, 'handlers'):  # exception-objects: try
                    for h in obj.handlers:
                        self.get_obj(h, import_val)
                    for f in obj.finalbody:
                        self.get_obj(f, import_val)
                    obj.handlers = []
                    obj.finalbody = []
        if type(obj) != ast.Import and type(obj) != ast.ImportFrom:  # import-objects: import, import from
            dump_ast = ast.dump(obj)
            for i in import_val:
                import_string = "'" + i + "'"
                if import_string in dump_ast:
                    self.preselected_ast_objects.append(obj)
                    break

    def get_ast_objects_containing_classes(self, classes):
        for s in self.preselected_ast_objects:
            dump_ast = ast.dump(s)
            for c in classes:
                c_edit_dump = "\'" + c + "\'"
                if c_edit_dump in dump_ast:
                    c_edit_unparse = c + "("
                    if c_edit_unparse in ast.unparse(s):
                        final_obj = {"class": c, "obj": s, "param variables": None}
                        self.final_ast_objects.append(final_obj)

    def get_param_variables(self, parent_ast_objects):
        global_assignments = []
        for obj in parent_ast_objects:
            if type(obj) == ast.Assign:
                global_assignments.append(obj)
        self.functions = sorted(self.functions, key=lambda d: d['line no end'])
        for obj in self.final_ast_objects:
            for func in self.functions:
                if func["line no"] <= obj["obj"].lineno:
                    if obj["obj"].lineno <= func["line no end"]:
                        relevant_assigns = []
                        for assign in func["assignments"]:
                            if assign.lineno < obj["obj"].lineno:
                                relevant_assigns.append(assign)
                            else:
                                break
                        obj["param variables"] = relevant_assigns
                        break
                else:
                    relevant_global_assigns = []
                    for assign in global_assignments:
                        if assign.lineno < obj["obj"].lineno:
                            relevant_global_assigns.append(assign)
                        else:
                            break
                    obj["param variables"] = relevant_global_assigns
                    break


def main():
    project = "test_projects/another_test_project.py"
    ml_lib = "sklearn"

    ast_objects = CodeObjects(ml_lib, project).get_objects()
    pprint(ast_objects, width=75)


if __name__ == "__main__":
    main()
