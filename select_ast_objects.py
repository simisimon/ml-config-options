import ast
from pprint import pprint
from class_scraper import sklearn_classes


def get_parent_ast_objects(prj):
    with open(prj, "r") as source:
        tree = ast.parse(source.read())
    obj = tree.body
    return obj


def get_import_val(obj):
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


def get_selected_ast_objects():
    for p in parent_ast_objects:
        get_obj(p)


def get_obj(obj):
    if hasattr(obj, 'body'):
        for b in obj.body:  # func, async func, class, with, async with, except handler
            get_obj(b)
        obj.body = []
        if hasattr(obj, 'orelse'):  # control-flow-objects: for, if, async for, while
            for o in obj.orelse:
                get_obj(o)
            obj.orelse = []
            if hasattr(obj, 'handlers'):  # exception-objects: try
                for h in obj.handlers:
                    get_obj(h)
                for f in obj.finalbody:
                    get_obj(f)
                obj.handlers = []
                obj.finalbody = []
    if type(obj) != ast.Import and type(obj) != ast.ImportFrom:  # import-objects: import, import from
        dump_ast = ast.dump(obj)
        for i in import_values:
            import_string = "'" + i + "'"
            if import_string in dump_ast:
                preselected_ast_objects.append(obj)
                break


def get_ast_objects_containing_classes():
    final_ast_objects = []
    for s in preselected_ast_objects:
        dump_ast = ast.dump(s)
        for c in classes:
            if c in dump_ast:
                final_ast_objects.append(s)
                break
    return final_ast_objects


project = "test_projects/another_test_project.py"
ml_lib = "sklearn"
classes = sklearn_classes

parent_ast_objects = get_parent_ast_objects(project)
import_values = get_import_val(parent_ast_objects)
print("import values: " + str(import_values))

preselected_ast_objects = []
get_selected_ast_objects()
for p in preselected_ast_objects:
    print("preselection: " + ast.unparse(p))
final_ast_objects = get_ast_objects_containing_classes()
for f in final_ast_objects:
    print("final selection: " + ast.unparse(f))
