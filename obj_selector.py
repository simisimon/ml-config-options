import ast
from pprint import pprint
import json

preselected_ast_objects = []


def read_json(ml_lib):
    with open(ml_lib + '.txt') as json_file:
        classes = json.load(json_file)
    return classes


def get_parent_ast_objects(prj):
    with open(prj, "r") as source:
        tree = ast.parse(source.read())
    obj = tree.body
    return obj


def get_import_val(obj, ml_lib):
    import_obj = []
    for o in obj:
        desc_nodes = list(ast.walk(o))
        for d in desc_nodes:
            if type(d) == ast.Import:
                for n in d.names:
                    if ml_lib in n.name:
                        if n.asname != None:
                            import_obj.append(n.asname)
                        else:
                            import_obj.append(n.name)
            elif type(d) == ast.ImportFrom:
                for n in d.names:
                    if ml_lib in d.module:
                        if n.asname != None:
                            import_obj.append(n.asname)
                        else:
                            import_obj.append(n.name)
    return import_obj


def preselect_ast_objects(parent_obj, import_val):
    for p in parent_obj:
        get_obj(p, import_val)


def get_obj(obj, import_val):
    preselected_obj = []
    if hasattr(obj, 'body'):
        for b in obj.body:  # func, async func, class, with, async with, except handler
            get_obj(b, import_val)
        obj.body = []
        if hasattr(obj, 'orelse'):  # control-flow-objects: for, if, async for, while
            for o in obj.orelse:
                get_obj(o, import_val)
            obj.orelse = []
            if hasattr(obj, 'handlers'):  # exception-objects: try
                for h in obj.handlers:
                    get_obj(h, import_val)
                for f in obj.finalbody:
                    get_obj(f, import_val)
                obj.handlers = []
                obj.finalbody = []
    if type(obj) != ast.Import and type(obj) != ast.ImportFrom:  # import-objects: import, import from
        dump_ast = ast.dump(obj)
        for i in import_val:
            import_string = "'" + i + "'"
            if import_string in dump_ast:
                preselected_ast_objects.append(obj)
                break
    return preselected_obj


def get_ast_objects_containing_classes(classes):
    final_obj = {}
    for s in preselected_ast_objects:
        dump_ast = ast.dump(s)
        for c in classes:
            if c in dump_ast:
                final_obj[s.lineno] = ast.unparse(s)
                break
    return final_obj


def main():
    project = "test_projects/another_test_project.py"
    ml_lib = "sklearn"

    classes = read_json(ml_lib)
    parent_ast_objects = get_parent_ast_objects(project)
    import_values = get_import_val(parent_ast_objects, ml_lib)
    preselect_ast_objects(parent_ast_objects, import_values)
    #for p in preselected_ast_objects:
     #   print("preselection: " + ast.unparse(p))

    final_ast_objects = {}
    final_ast_objects = get_ast_objects_containing_classes(classes)
    pprint(final_ast_objects, width=250)


if __name__ == "__main__":
    main()
