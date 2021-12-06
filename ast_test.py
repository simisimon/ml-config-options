import ast
from operator import attrgetter
from pprint import pprint

# https://stackoverflow.com/questions/33506902/python-extracting-editing-all-constants-involved-in-a-function-via-ast
project = "test_projects/sklearn_birch.py"
final_vars = {}

def get_objects(project):
    with open(project, "r") as source:
        tree = ast.parse(source.read())

    root = ast.parse(tree)
    objects = root.body
    return objects

def extract(objects):
    for obj in objects:
        obj_type = type(obj)
        extract_vars(obj, obj_type)

    return final_vars

def extract_vars(obj, obj_type):
    if obj_type == ast.Assign:
        extract_assigns(obj)
    elif obj_type == ast.FunctionDef:
        extract_func(obj)

def extract_assigns(obj):
    assign_vars = []
    #print(type(obj.value))
    if type(obj.targets[0]) == ast.Tuple:
        assign_vars.extend(obj.targets[0].elts)
    else:
        assign_vars.append(obj.targets[0])

    for assign in assign_vars:
        if type(obj.value) == ast.Tuple:
            values = list(map(attrgetter('n'), obj.value.elts))
        elif type(obj.value) == ast.List:
            values = []
            for o in obj.value.elts:
                #if type(o.elts) == ast.
                list_values = []
                for v in o.elts:
                    if type(v) == ast.UnaryOp:
                        if type(v.op) == ast.USub:
                            list_values.append(-v.operand.n)
                    else:
                        list_values.append(v.n)
                values.append(list_values)
        elif type(obj.value) == ast.Call:
            values = {}
            values[obj.value.func.id] = []
            for param in obj.value.args:
                values[obj.value.func.id].append(param.n)
            for param in obj.value.keywords:
                values[obj.value.func.id].append(param.arg + "=" + str(param.value.n))
        elif type(obj.value) == ast.BinOp:
            values = "BinOp???" #ast.dump(ast.parse('x ** n + y ** n + k', mode='eval'), indent=4))
        else:
            values = obj.value.n



        final_vars[assign.id] = values

def extract_func(obj):
    assigns = obj.body
    for assign in assigns:
        if type(assign) == ast.Assign:
            extract_assigns(assign)
        else:
            print("return?!")

objects = get_objects(project)
result = extract(objects)
pprint(result)

#with open(project, "r") as source:
 #   tree = ast.parse(source.read())

#print(ast.dump(tree, indent=4))


#gg = ast.iter_child_nodes(node)    # iterate over child nodes









