import ast
from pprint import pprint


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
        get_val(p)


def get_val(obj):
    if type(obj) in category1:  # consider body
        get_category_1(obj)
    elif type(obj) in category2:  # consider body and orelse
        get_category_2(obj)
    elif type(obj) == ast.Try:  # consider body, orelse, handlers and finalbody
        get_try(obj)
    elif type(obj) in category3: # not relevant at all
        pass
    else:  # nothing to consider
        get_category_4(obj)


def get_category_1(obj):
    for b in obj.body:
        get_val(b)


def get_category_2(obj):
    for b in obj.body:
        get_val(b)
    for o in obj.orelse:
        get_val(o)


def get_try(obj):
    for b in obj.body:
        get_val(b)
    for o in obj.orelse:
        get_val(o)
    for h in obj.handlers:
        get_val(h)
    for f in obj.finalbody:
        get_val(f)


def get_category_4(obj):
    dump_ast = ast.dump(obj)
    for i in import_values:
        import_string = "'" + i + "'"
        if import_string in dump_ast:
            selected_ast_objects.append(obj)


project = "test_projects/another_test_project.py"
ml_lib = "sklearn"

parent_ast_objects = get_parent_ast_objects(project)
import_values = get_import_val(parent_ast_objects)
pprint(import_values)

category1 = [ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.With, ast.AsyncWith, ast.ExceptHandler]
category2 = [ast.For, ast.AsyncFor, ast.While, ast.If]
category3 = [ast.Import, ast.ImportFrom]
selected_ast_objects = []
get_selected_ast_objects()
pprint(selected_ast_objects)
for p in selected_ast_objects:
    pprint(ast.unparse(p), width=300)


# missing: iterate through specific parent_obj at category 1-2 and try
# next Step: Iterate over selected_ast_objects to extract objects with ml-lib classes