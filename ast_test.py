import ast
from operator import attrgetter
from pprint import pprint

# https://stackoverflow.com/questions/33506902/python-extracting-editing-all-constants-involved-in-a-function-via-ast
project = "test_projects/sklearn_lin_reg.py"
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
    # print(type(obj.value))
    if type(obj.targets[0]) == ast.Tuple:
        if type(obj.value) == ast.Tuple or type(obj.value) == ast.List:
            if len(obj.targets[0].elts) == len(obj.value.elts):
                assign_vars.extend(obj.targets[0].elts)
        else:
            assign_vars.extend(obj.targets[0].elts)
    else:
        assign_vars.append(obj.targets[0])

    for assign in assign_vars:
        ast_type = type(obj.value)
        values = obj_type[ast_type](obj.value)
        if type(obj.targets[0]) == ast.Tuple:
            if type(obj.value) == ast.Tuple or type(obj.value) == ast.List:
                final_vars[assign.id] = values[assign_vars.index(assign)]
            else:
                final_vars[assign.id] = values
        else:
            final_vars[assign.id] = values


def extract_func(obj):
    assigns = obj.body
    for assign in assigns:
        if type(assign) == ast.Assign:
            extract_assigns(assign)
        else:
            print("return?!")


def extract_tuple(obj_val):
    values = []
    for o in obj_val.elts:
        ast_type = type(o)
        values.append(obj_type[ast_type](o))
    return values


def extract_list(obj_val):
    values = []
    for o in obj_val.elts:
        ast_type = type(o)
        values.append(obj_type[ast_type](o))

        if type(o) == ast.List:
            list_values = []
            for v in o.elts:
                ast_type = type(v)
                list_values.append(obj_type[ast_type](v))
            values.append(list_values)
    return values


def extract_call(obj_val):
    ast_type = type(obj_val.func)
    values = obj_type[ast_type](obj_val.func)

    values += "("
    for param in obj_val.args:
        if values[-1] != "(":
            values += ", "
        ast_type = type(param)
        param_value = obj_type[ast_type](param)
        values += str(param_value)
    for param in obj_val.keywords:
        if values[-1] != "(":
            values += ", "
        ast_type = type(param.value)
        param_value = obj_type[ast_type](param.value)
        values += param.arg + "=" + str(param_value)
    values += ")"
    return values


def extract_bin_op(obj_val):
    values = "BinOp???"  # ast.dump(ast.parse('x ** n + y ** n + k', mode='eval'), indent=4))
    return values


def extract_constant(obj_val):
    values = obj_val.n
    return values


def extract_unary_op(obj_val):
    if type(obj_val.op) == ast.USub:
        return -obj_val.operand.n
    if type(obj_val.op) == ast.UAdd:
        return +obj_val.operand.n
    if type(obj_val.op) == ast.Not:
        return not obj_val.operand.n
    if type(obj_val.op) == ast.Invert:
        return ~obj_val.operand.n


def extract_name(obj_val):
    values = obj_val.id
    return values


def extract_attr(obj_val):
    ast_type = type(obj_val.value)
    values = obj_type[ast_type](obj_val.value)
    values += "." + obj_val.attr
    return values

obj_type = {ast.Tuple: extract_tuple,
            ast.List: extract_list,
            ast.Call: extract_call,
            ast.BinOp: extract_bin_op,
            ast.Constant: extract_constant,
            ast.UnaryOp: extract_unary_op,
            ast.Name: extract_name,
            ast.Attribute: extract_attr}

objects = get_objects(project)
result = extract(objects)
pprint(result)

#with open(project, "r") as source:
  #  tree = ast.parse(source.read())

#print(ast.dump(tree, indent=4))


# gg = ast.iter_child_nodes(node)    # iterate over child nodes
