import ast
from operator import attrgetter
from pprint import pprint
import re

# https://stackoverflow.com/questions/33506902/python-extracting-editing-all-constants-involved-in-a-function-via-ast
project = "test_projects/sklearn_lin_reg.py"
final_vars = [] ##dict


def get_objects(project):
    with open(project, "r") as source:
        tree = ast.parse(source.read())

    root = ast.parse(tree)
    objects = root.body

    return objects


def extract(objects):
    for obj in objects:
        obj_type[type(obj)](obj)
    return final_vars


def extract_assigns(obj):
    if type(obj.targets[0]) != ast.Tuple: #Case 1: Normalfall
        values = obj_type[type(obj.value)](obj.value)
        assign_vars = extraxt_assign_vars(obj.targets[0])
        final_vars.append(assign_vars + " = " + str(values))

    elif type(obj.value) != ast.Tuple and type(obj.value) != ast.List: #Case 2: Zusammenfassung
        values = obj_type[type(obj.value)](obj.value)
        if type(obj.value) != ast.Constant: #Tupel = Konstante syntaktisch falsch und wird geskippt
            assign_vars = ""
            for assign in obj.targets[0].elts:
                assign_vars += str((extraxt_assign_vars(assign))) + ", "
            final_vars.append(assign_vars[:-2] + " = " + str(values))
        else:
            print("Syntaktisch nicht zulässig")

    elif len(obj.targets[0].elts) == len(obj.value.elts): #Case 3: Mehrere Assignments
        targets = obj.targets[0].elts
        for assign in targets:
            assign_var = extraxt_assign_vars(assign)
            value = obj.value.elts[targets.index(assign)]
            value = obj_type[type(value)](value)
            final_vars.append(assign_var + " = " + str(value))

    else:
        print("NEUER FALL!!!!!! bzw. Falsches Assignment") #kein Assignment


def extraxt_assign_vars(assign_obj):
    ast_type = type(assign_obj)
    values = obj_type[ast_type](assign_obj)
    #if type(assign_obj) == ast.Attribute:
     #   assign_id = extract_attr(assign_obj)
    #elif type(assign_obj) == ast.Tuple:
     #   assign_id = extract_tuple(assign_obj)
    #else:
     #   assign_id = assign_obj.id
    return values


def extract_expr(obj):
    if type(obj.value) == ast.Constant:
        print("docstring?")
    else:
        ast_type = type(obj.value)
        values = obj_type[ast_type](obj.value)
        final_vars.append(values)


def dummy(obj):
    pass


def extract_func(obj):
    assigns = obj.body
    for assign in assigns:
        if type(assign) == ast.Assign:
            extract_assigns(assign)
        else:
            print("return?!")


def extract_tuple(obj_val):

    values = "("
    for o in obj_val.elts:
        ast_type = type(o)
        values += str(obj_type[ast_type](o)) + ", "
    values = values[:-2] + ")"
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
    ast_type = type(obj_val.left)
    values = obj_type[ast_type](obj_val.left)
    if type(obj_val.op) == ast.Add:
        values += " + "
    elif type(obj_val.op) == ast.Sub:
        values += " - "
    elif type(obj_val.op) == ast.Mult:
        values += " * "
    elif type(obj_val.op) == ast.Div:
        values += " / "
    elif type(obj_val.op) == ast.FloorDiv:
        values += " // "
    elif type(obj_val.op) == ast.Mod:
        values += " % "
    elif type(obj_val.op) == ast.Pow:
        values += " ** "
    elif type(obj_val.op) == ast.LShift:
        values += " << "
    elif type(obj_val.op) == ast.RShift:
        values += " >> "
    elif type(obj_val.op) == ast.BitOr:
        values += " | "
    elif type(obj_val.op) == ast.BitXor:
        values += " ^ "
    elif type(obj_val.op) == ast.BitAnd:
        values += " & "
    elif type(obj_val.op) == ast.MatMult:
        values += " @ "

    ast_type = type(obj_val.right)
    values += obj_type[ast_type](obj_val.right)

    return values


def extract_constant(obj_val):
    values = obj_val.n
    return values


def extract_unary_op(obj_val):
    if type(obj_val.op) == ast.USub:
        return -obj_val.operand.n
    elif type(obj_val.op) == ast.UAdd:
        return +obj_val.operand.n
    elif type(obj_val.op) == ast.Not:
        return not obj_val.operand.n
    elif type(obj_val.op) == ast.Invert:
        return ~obj_val.operand.n


def extract_name(obj_val):
    values = obj_val.id
    return values


def extract_attr(obj_val):
    ast_type = type(obj_val.value)
    values = obj_type[ast_type](obj_val.value)
    values += "." + obj_val.attr
    return values


def extract_dict(obj_val):
    values = {}
    count = 0
    for key in obj_val.keys:
        ast_type = type(key)
        k = obj_type[ast_type](key)

        v = obj_val.values[count]
        ast_type = type(v)
        v = obj_type[ast_type](v)

        values[k] = v
        count += 1

    return values

def extract_subscript(obj_val):
    ast_type = type(obj_val.value)
    values = obj_type[ast_type](obj_val.value) + "["
    ast_type = type(obj_val.slice)
    values += obj_type[ast_type](obj_val.slice) + "]"
    return values


def extract_slice(obj_val):
    values = ""
    lower = obj_val.lower
    upper = obj_val.upper
    step = obj_val.step
    if lower != None:
        ast_type = type(lower)
        values += str(obj_type[ast_type](lower))
    values += ":"
    if upper != None:
        ast_type = type(upper)
        values += str(obj_type[ast_type](upper))
    if step != None:
        ast_type = type(step)
        values += ":" + str(obj_type[ast_type](step))
    return values


def extract_ext_slice(obj_val):
    values = ""
    for d in obj_val.dims:
        ast_type = type(d)
        values += str(obj_type[ast_type](d)) + ", "
    return values[:-2]


def extract_index(obj_val):
    ast_type = type(obj_val.value)
    values = obj_type[ast_type](obj_val.value)
    return values


def extract_with(obj_val):
    values = ""
    for item in obj_val.items:
        ast_type = type(item)
        final_vars.append(obj_type[ast_type](item))

    for v in obj_val.body:
        ast_type = type(v)
        val_temp = obj_type[ast_type](v)
        if val_temp != None:
            values += val_temp
    return values


def extract_withitem(obj_val):
    values = "with "
    ast_type = type(obj_val.context_expr)
    values += obj_type[ast_type](obj_val.context_expr)

    if obj_val.optional_vars != None:
        ast_type = type(obj_val.optional_vars)
        values += " as " + obj_type[ast_type](obj_val.optional_vars)
    return values + ":"


obj_type = {ast.Expr: extract_expr,
            ast.Assign: extract_assigns,
            ast.FunctionDef: extract_func,
            ast.Tuple: extract_tuple,
            ast.List: extract_list,
            ast.Call: extract_call,
            ast.BinOp: extract_bin_op,
            ast.Constant: extract_constant,
            ast.UnaryOp: extract_unary_op,
            ast.Name: extract_name,
            ast.Attribute: extract_attr,
            ast.Dict: extract_dict,
            ast.Subscript: extract_subscript,
            ast.Slice: extract_slice,
            ast.ExtSlice: extract_ext_slice,
            ast.Index: extract_index,
            ast.With: extract_with,
            ast.withitem: extract_withitem,
            ast.Import: dummy,
            ast.ImportFrom: dummy,
            ast.Assert: dummy}

objects = get_objects(project)
result = extract(objects)
pprint(result)

#with open(project, "r") as source:
  #  tree = ast.parse(source.read())

#tree2 = ast.parse(open(project).read())

#tree
#pprint(ast.dump(tree))


# gg = ast.iter_child_nodes(node)    # iterate over child nodes
