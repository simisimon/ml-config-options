import ast
from operator import attrgetter
from pprint import pprint
import re

# https://stackoverflow.com/questions/33506902/python-extracting-editing-all-constants-involved-in-a-function-via-ast
project = "test_projects/sklearn_birch.py"
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
        values = get_values(obj.value)
        assign_vars = get_values(obj.targets[0])
        final_vars.append(assign_vars + " = " + str(values))

    elif type(obj.value) != ast.Tuple and type(obj.value) != ast.List: #Case 2: Zusammenfassung
        values = get_values(obj.value)
        if type(obj.value) != ast.Constant: #Tupel = Konstante syntaktisch falsch und wird geskippt
            assign_vars = ""
            for assign in obj.targets[0].elts:
                assign_vars += str(get_values(assign)) + ", "
            final_vars.append(assign_vars[:-2] + " = " + str(values))
        else:
            print("Syntaktisch nicht zulässig")

    elif len(obj.targets[0].elts) == len(obj.value.elts): #Case 3: Mehrere Assignments
        targets = obj.targets[0].elts
        for assign in targets:
            assign_var = get_values(assign)
            value = obj.value.elts[targets.index(assign)]
            value = get_values(value)
            final_vars.append(assign_var + " = " + str(value))

    else:
        print("NEUER FALL!!!!!! bzw. Falsches Assignment") #kein Assignment


def get_values(obj_value):
    ast_type = type(obj_value)
    values = obj_type[ast_type](obj_value)
    return values


def extract_expr(obj):
    if type(obj.value) == ast.Constant:
        print("docstring?")
    else:
        values = get_values(obj.value)
        final_vars.append(values)


def dummy(obj):
    print("dummy: " + str(type(obj)))
    pass


def extract_func(obj):
    body = obj.body
    return_value = ""
    for e in body:
        values = get_values(e)
        if type(values) == str:
            if values.startswith("return "):
                args = extract_args(obj.args)
                return_value = "def: " + obj.name + args + ": " + values
                final_vars.append(return_value)
    if return_value == "":
        args = extract_args(obj.args)
        values = "def: " + obj.name + args
        final_vars.append(values)


def extract_args(args):
    arg_dict = {}
    #order: 1. posonylargs(x) + /, 2. args(x), 1.5 default(=1), 3. *, 4.vararg(x), 5. kwonlyargs(x), 6. kwdefault(=1), 7. **kwarg
    if len(args.posonlyargs) > 0:
        for arg in args.posonlyargs: #parameter before /
            annotation = ""
            if arg.annotation != None:
                annotation = ":" + str(get_values(arg.annotation))
            arg_dict[get_position(arg)] = arg.arg + annotation
        # place "/" at the right spot
        if len(args.defaults) == 0:
            pos = float(get_position(args.posonlyargs[len(args.posonlyargs) - 1]))
            arg_dict[str(pos + 0.0001)] = "/"
        elif len(args.args) == 0:
            pos = float(get_position(args.defaults[len(args.defaults) - 1]))
            arg_dict[str(pos + 0.0001)] = "/"
        else:
            pos = float(get_position(args.args[0]))
            arg_dict[str(pos - 0.0001)] = "/"

    for arg in args.args: #parameter
        annotation = ""
        if arg.annotation != None:
            annotation = ":" + str(get_values(arg.annotation))
        arg_dict[get_position(arg)] = arg.arg + annotation

    for arg in args.defaults: #default of args and posonlyargs
        arg_dict[get_position(arg)] = "=" + str(get_values(arg))

    if args.vararg != None:  #*-Parameter
        annotation = ""
        if args.vararg.annotation != None:
            annotation = ":" + str(get_values(args.vararg.annotation))
        arg_dict[get_position(args.vararg)] = "*" + args.vararg.arg + annotation

    for arg in args.kwonlyargs:  #parameter after *-parameter
        annotation = ""
        if arg.annotation != None:
            annotation = ":" + str(get_values(arg.annotation))
        arg_dict[get_position(arg)] = arg.arg + annotation

    if len(args.kw_defaults) > 0:   #default of kwonlyargs
        if args.vararg == None:
            pos = float(get_position(args.kwonlyargs[0]))
            arg_dict[str(pos-0.0001)] = "*"  #to place "*" at the right spot

        for arg in args.kw_defaults:
            if arg != None:
                arg_dict[get_position(arg)] = "=" + str(get_values(arg))

    if args.kwarg != None: #**-parameter
        annotation = ""
        if args.kwarg.annotation != None:
            annotation = ":" + str(get_values(args.kwarg.annotation))
        arg_dict[get_position(args.kwarg)] = "**" + args.kwarg.arg + annotation

    arg_dict = dict(sorted(arg_dict.items()))
    param = ""
    for val in arg_dict.values():
        if val.startswith("="):
            param = param[:-2]
        param += val + ", "
    param = "(" + param[:-2] + ")"
    return param


def get_position(obj):
    line_no = str(obj.lineno)
    col_no = str(obj.col_offset).zfill(3)
    return line_no + "." + col_no

def extract_tuple(obj_val):
    values = "("
    for o in obj_val.elts:
        values += str(get_values(o)) + ", "
    values = values[:-2] + ")"
    return values


def extract_list(obj_val):
    values = []
    for o in obj_val.elts:
        values.append(get_values(o))

        if type(o) == ast.List:
            list_values = []
            for v in o.elts:
                list_values.append(get_values(v))
            values.append(list_values)
    return values


def extract_call(obj_val):
    values = get_values(obj_val.func)

    values += "("
    for param in obj_val.args:
        if values[-1] != "(":
            values += ", "
        param_value = get_values(param)
        values += str(param_value)
    for param in obj_val.keywords:
        if values[-1] != "(":
            values += ", "
        param_value = get_values(param.value)
        values += param.arg + "=" + str(param_value)
    values += ")"
    return values


def extract_bin_op(obj_val):
    values = get_values(obj_val.left)
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

    values += str(get_values(obj_val.right))
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
    values = get_values(obj_val.value)
    values += "." + obj_val.attr
    return values


def extract_dict(obj_val):
    values = {}
    count = 0
    for key in obj_val.keys:
        k = get_values(key)

        v = obj_val.values[count]
        v = get_values(v)

        values[k] = v
        count += 1

    return values


def extract_subscript(obj_val):
    values = get_values(obj_val.value) + "["
    values += get_values(obj_val.slice) + "]"
    return values


def extract_slice(obj_val):
    values = ""
    lower = obj_val.lower
    upper = obj_val.upper
    step = obj_val.step
    if lower != None:
        values += str(get_values(lower))
    values += ":"
    if upper != None:
        values += str(get_values(upper))
    if step != None:
        values += ":" + str(get_values(step))
    return values


def extract_ext_slice(obj_val):
    values = ""
    for d in obj_val.dims:
        values += str(get_values(d)) + ", "
    return values[:-2]


def extract_index(obj_val):
    values = get_values(obj_val.value)
    return values


def extract_with(obj_val):
    values = ""
    for item in obj_val.items:
        final_vars.append(get_values(item))

    for v in obj_val.body:
        val_temp = get_values(v)
        if val_temp != None:
            values += val_temp
    return values


def extract_withitem(obj_val):
    values = "with "
    values += get_values(obj_val.context_expr)

    if obj_val.optional_vars != None:
        values += " as " + get_values(obj_val.optional_vars)
    return values + ":"

def extract_if(obj_val):
    #ignore condition
    for e in obj_val.body:
        get_values(e)

def extract_return(obj_val):
    values = "return " + str(get_values(obj_val.value))
    return values

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
            ast.Assert: dummy,
            ast.If: extract_if,
            ast.Return: extract_return}

objects = get_objects(project)
result = extract(objects)
pprint(result)

#with open(project, "r") as source:
  #  tree = ast.parse(source.read())

#tree2 = ast.parse(open(project).read())

#tree
#pprint(ast.dump(tree))


# gg = ast.iter_child_nodes(node)    # iterate over child nodes
