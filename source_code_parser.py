import ast
from operator import attrgetter
from pprint import pprint
import re

project = "test_projects/another_test_project.py"
final_vars = {}

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


def get_values(obj_value):
    ast_type = type(obj_value)
    values = obj_type[ast_type](obj_value)
    return values

"""ast-handling"""
def extract_func(obj):
    if type(obj) == ast.FunctionDef:
        func_type = "def: "
    else:
        func_type = "async def: "
    body = obj.body
    return_value = ""
    for b in body:
        values = get_values(b)
        if type(values) == str:
            if values.startswith("return "):
                args = extract_args(obj.args)
                return_value = func_type + obj.name + args + ": " + values
                final_vars[b.lineno] = return_value
    if return_value == "":
        args = extract_args(obj.args)
        values = func_type + obj.name + args
        final_vars[obj.lineno] = values
    for d in obj.decorator_list:
        final_vars[d.lineno] = "@" + str(get_values(d))


def extract_class(obj):
    value = "class " + obj.name + "("
    for b in obj.bases:
        value += get_values(b) + ", "
    final_vars[obj.lineno] = value[:-2] + ")"
    for b in obj.body:
        get_values(b)


def extract_return(obj_val):
    values = "return " + str(get_values(obj_val.value))
    return values


def extract_assigns(obj):
    if type(obj.targets[0]) != ast.Tuple: #Case 1: Normalfall
        values = get_values(obj.value)
        assign_vars = get_values(obj.targets[0])
        final_vars[obj.lineno] = assign_vars + " = " + str(values)


    elif type(obj.value) != ast.Tuple and type(obj.value) != ast.List: #Case 2: Zusammenfassung
        values = get_values(obj.value)
        if type(obj.value) != ast.Constant: #Tupel = Konstante syntaktisch falsch und wird geskippt
            assign_vars = ""
            for assign in obj.targets[0].elts:
                assign_vars += str(get_values(assign)) + ", "
            final_vars[obj.lineno] = assign_vars[:-2] + " = " + str(values)
        else:
            print("Syntaktisch nicht zulässig")

    elif len(obj.targets[0].elts) == len(obj.value.elts): #Case 3: Mehrere Assignments
        targets = obj.targets[0].elts
        for assign in targets:
            assign_var = get_values(assign)
            value = obj.value.elts[targets.index(assign)]
            value = get_values(value)
            final_vars[float(str(obj.lineno) + "." + str(targets.index(assign)))] = assign_var + " = " + str(value)

    else:
        print("NEUER FALL!!!!!! bzw. Falsches Assignment") #kein Assignment


def extract_aug_assign(obj):
    value = get_values(obj.target)
    value += get_operator(obj.op) + "= "
    value += str(get_values(obj.value))
    final_vars[obj.lineno] = value


def extract_ann_assign(obj):
    value = get_values(obj.target) + ": "
    value += get_values(obj.annotation)
    if obj.value != None:
        value += " = " + str(get_values(obj.value))
    final_vars[obj.lineno] = value

def extract_for(obj):
    if type(obj) == ast.For:
        values = "for "
    else:
        values = "async for "
    values += get_values(obj.target) + " in "
    values += get_values(obj.iter) + ":"
    for b in obj.body:
        get_values(b)
    final_vars[obj.lineno] = values
    return ""


def extract_while(obj):
    values = "while ("
    values += str(get_values(obj.test)) + "):"
    final_vars[obj.lineno] = values
    for b in obj.body:
        get_values(b)
    if len(obj.orelse) > 0:
        for o in obj.orelse:
            get_values(o)



def extract_if(obj_val):
    #ignore condition
    for b in obj_val.body:
        get_values(b)

    for b in obj_val.orelse:
        get_values(b)


def extract_with(obj):
    if type(obj) == ast.AsyncWith:
        values = "async "
    else:
        values = ""
    for item in obj.items:
        final_vars[obj.lineno] = (get_values(item))

    for v in obj.body:
        val_temp = get_values(v)
        if val_temp != None:
            values += val_temp
    return values


def extract_raise(obj):
    values = "raise "
    values += str(get_values(obj.exc))
    if obj.cause != None:
        values += " from " + str(get_values(obj.cause))
    final_vars[obj.lineno] = values


def extract_try(obj):
    values = ""
    for b in obj.body:
        get_values(b)
    for h in obj.handlers:
        get_values(h)
    for o in obj.orelse:
        get_values(o)
    for f in obj.finalbody:
        get_values(f)


def extract_global(obj):
    values = "global "
    for n in obj.names:
        values += n + ", "
    values = values[:-2]
    final_vars[obj.lineno] = values


def extract_nonlocal(obj):
    values = "nonlocal "
    for n in obj.names:
        values += n + ", "
    values = values[:-2]
    final_vars[obj.lineno] = values


def extract_expr(obj):
    if type(obj.value) == ast.Constant:
        print("docstring?")
    else:
        values = get_values(obj.value)
        final_vars[obj.lineno] = values


def extract_bool_op(obj):
    values = ""
    for v in obj.values:
        values += get_values(v)
        if obj.values.index(v) + 1 != len(obj.values):
            if type(obj.op) == ast.And:
                values += " and "
            elif type(obj.op) == ast.Or:
                values += " or "
    return values


def extract_named_expr(obj):
    value = str(get_values(obj.target)) + " := " + str(get_values(obj.value))
    return value


def extract_bin_op(obj_val):
    values = str(get_values(obj_val.left))
    values += get_operator(obj_val.op) + " "
    values += str(get_values(obj_val.right))
    return values


def extract_unary_op(obj_val):
    if type(obj_val.op) == ast.USub:
        return "-" + str(get_values(obj_val.operand))
    elif type(obj_val.op) == ast.UAdd:
        return "+" + str(get_values(obj_val.operand))
    elif type(obj_val.op) == ast.Not:
        return "not " + str(get_values(obj_val.operand))
    elif type(obj_val.op) == ast.Invert:
        return "~" + str(get_values(obj_val.operand))


def extract_lambda(obj_val):
    value = "lambda"
    args = extract_args(obj_val.args)[1:-1]
    if args != "":
        value += " " + args
    value += ": " + str(get_values(obj_val.body))
    return value


def extract_if_exp(obj):
    value = str(get_values(obj.body))
    value += " if " + str(get_values(obj.test))
    value += " else " + str(get_values(obj.orelse))
    return value


def extract_dict(obj_val):
    values = "{"
    count = 0
    for key in obj_val.keys:
        values += str(get_values(key)) + ": "
        v = obj_val.values[count]
        if type(v) != ast.List:
            values += str(get_values(v)) + ", "
        else:
            values += "["
            for e in get_values(v):
                values += str(e) + ", "
            values = values[:-2] + "]" + ", "
        count += 1

    if len(obj_val.keys) > 0:
        values = values[:-2]
    values += "}"
    return values


def extract_set(obj):
    value = "{"
    for e in obj.elts:
        value += str(get_values(e)) + ", "
    value = value[:-2] + "}"
    return value


def extract_list_comp(obj):
    values = "[" + str(get_values(obj.elt))
    for g in obj.generators:
        values += str(get_values(g))
    values += "]"
    return values


def extract_set_comp(obj):
    values = "{" + str(get_values(obj.elt))
    for g in obj.generators:
        values += str(get_values(g))
    values += "}"
    return values


def extract_dict_comp(obj):
    values = "{" + str(get_values(obj.key)) + ":"
    values += str(get_values(obj.value))
    for g in obj.generators:
        values += str(get_values(g))
    values += "}"
    return values


def extract_generator_exp(obj):
    value = get_values(obj.elt)
    for g in obj.generators:
        value += get_values(g)
    return value


def extract_await(obj):
    value = "await "
    value += str(get_values(obj.value))
    return value


def extract_yield(obj):
    value = "yield " + get_values(obj.value)
    return value


def extract_yield_from(obj):
    value = "yield from " + get_values(obj.value)
    return value


def extract_compare(obj_val):
    values = str(get_values(obj_val.left))
    for c in obj_val.comparators:
        values += " " + get_cmpop(obj_val.ops[obj_val.comparators.index(c)]) + " " + str(get_values(c))
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
        values += str(param.arg) + "=" + str(param_value)
    values += ")"
    return values


def extract_joined_str(obj_val):
    values = "f" + "'"
    for v in obj_val.values:
        if type(v) != ast.FormattedValue:
            values += get_values(v)[1:-1]
        else:
            values += get_values(v)
    values = values + "'"
    return values


def extract_formatted_val(obj_val):
    value = "{" + get_values(obj_val.value)
    if obj_val.format_spec != None:
        value += ":" + get_values(obj_val.format_spec)[2:-1]
    value += "}"
    return value


def extract_constant(obj_val):
    if type(obj_val.n) == str:
        values = "'" + obj_val.n + "'"
    else:
        values = obj_val.n
    return values


def extract_attr(obj_val):
    values = get_values(obj_val.value)
    values += "." + obj_val.attr
    return values


def extract_subscript(obj_val):
    values = get_values(obj_val.value) + "["
    values += str(get_values(obj_val.slice)) + "]"
    return values


def extract_starred(obj):
    values = "*" + str(get_values(obj.value))
    return values


def extract_name(obj_val):
    values = obj_val.id
    return values


def extract_list(obj_val):
    values = "["
    for o in obj_val.elts:
        values += str(get_values(o)) + ", "
    if len(obj_val.elts) > 0:
        values = values[:-2]
    values += "]"
    return values


def extract_tuple(obj_val):
    values = "" #"("
    for o in obj_val.elts:
        values += str(get_values(o)) + ", "
    values = values[:-2] #+ ")"
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


def get_cmpop(op):
    if type(op) == ast.Eq:
        return "=="
    elif type(op) == ast.NotEq:
        return "!="
    elif type(op) == ast.Lt:
        return "<"
    elif type(op) == ast.LtE:
        return "<="
    elif type(op) == ast.Gt:
        return ">"
    elif type(op) == ast.GtE:
        return ">="
    elif type(op) == ast.Is:
        return "is"
    elif type(op) == ast.IsNot:
        return "is not"
    elif type(op) == ast.In:
        return "in"
    elif type(op) == ast.NotIn:
        return "not in"


def extract_comprehension(obj):
    value = " for " + str(get_values(obj.target))
    value += " in " + str(get_values(obj.iter))
    for i in obj.ifs:
        value += " if " + str(get_values(i))
    return value


def extract_except_handler(obj):
    for b in obj.body:
        get_values(b)


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


def extract_withitem(obj_val):
    values = "with "
    values += get_values(obj_val.context_expr)

    if obj_val.optional_vars != None:
        values += " as " + get_values(obj_val.optional_vars)
    return values + ":"


"""ast-handling helping functions"""
def dummy(obj):
    print(str(obj.lineno) + ": dummy " + str(type(obj)))


def extract_ext_slice(obj_val):
    values = ""
    for d in obj_val.dims:
        values += str(get_values(d)) + ", "
    return values[:-2]


def get_operator(op):
    if type(op) == ast.Add:
        return " +"
    elif type(op) == ast.Sub:
        return " -"
    elif type(op) == ast.Mult:
        return " *"
    elif type(op) == ast.Div:
        return " /"
    elif type(op) == ast.FloorDiv:
        return " //"
    elif type(op) == ast.Mod:
        return " %"
    elif type(op) == ast.Pow:
        return " **"
    elif type(op) == ast.LShift:
        return " <<"
    elif type(op) == ast.RShift:
        return " >>"
    elif type(op) == ast.BitOr:
        return " |"
    elif type(op) == ast.BitXor:
        return " ^"
    elif type(op) == ast.BitAnd:
        return " &"
    elif type(op) == ast.MatMult:
        return " @"


def extract_index(obj_val):
    values = get_values(obj_val.value)
    return values


def get_position(obj):
    line_no = str(obj.lineno)
    col_no = str(obj.col_offset).zfill(3)
    return line_no + "." + col_no


obj_type = {ast.BoolOp: extract_bool_op,
            ast.Expr: extract_expr,
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
            ast.Return: extract_return,
            ast.ClassDef: extract_class,
            ast.For: extract_for,
            ast.Compare: extract_compare,
            ast.JoinedStr: extract_joined_str,
            ast.FormattedValue: extract_formatted_val,
            ast.Pass: dummy,
            ast.Break: dummy,
            ast.Continue: dummy,
            ast.Delete: dummy,
            ast.Lambda: extract_lambda,
            ast.AugAssign: extract_aug_assign,
            ast.AnnAssign: extract_ann_assign,
            ast.IfExp: extract_if_exp,
            ast.GeneratorExp: extract_generator_exp,
            ast.comprehension: extract_comprehension,
            ast.ListComp: extract_list_comp,
            ast.While: extract_while,
            ast.Yield: extract_yield,
            ast.Starred: extract_starred,
            ast.Try: extract_try,
            ast.ExceptHandler: extract_except_handler,
            ast.Raise: extract_raise,
            ast.Global: extract_global,
            ast.Nonlocal: extract_nonlocal,
            ast.YieldFrom: extract_yield_from,
            ast.AsyncFunctionDef: extract_func,
            ast.Await: extract_await,
            ast.AsyncFor: extract_for,
            ast.AsyncWith: extract_with,
            ast.DictComp: extract_dict_comp,
            ast.Set: extract_set,
            ast.SetComp: extract_set_comp,
            ast.NamedExpr: extract_named_expr
            }


objects = get_objects(project)
result = extract(objects)
pprint(result)

#with open(project, "r") as source:
  #  tree = ast.parse(source.read())

#tree2 = ast.parse(open(project).read())

#tree
#pprint(ast.dump(tree))


# gg = ast.iter_child_nodes(node)    # iterate over child nodes
