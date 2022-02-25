import ast
import json

from obj_selector import TorchObjects, SklearnObjects
from pprint import pprint


def get_parameters(objects):
    objects_with_prm = []
    for obj in objects:
        obj_code = ast.unparse(obj["object"])
        class_string = "{0}(".format(obj["class alias"])
        indices = [i for i in range(len(obj_code)) if obj_code.startswith(class_string, i)]
        for index in indices.copy():
            if index != 0 and (obj_code[index - 1].isalnum() or obj_code[index - 1] == "."):
                indices.remove(index)
        indices.reverse()

        """check if same class occurs multiple times in obj"""
        if len(indices) > 1:
            for i in range(len(indices)):
                for j in range(len(indices)):
                    if i != j:
                        start = indices[j]
                        stop = start + len(obj["class alias"])
                        obj_code = "".join((obj_code[:start], "temp", obj_code[stop:]))
                try:
                    new_obj = {"class": obj["class"], "class alias": obj["class alias"], "object": ast.parse(obj_code).body[0],
                                "parameter variables": obj["parameter variables"]}
                except:
                    new_obj = {"class": obj["class"], "class alias": obj["class alias"], "object": ast.parse(obj_code + "[]").body[0],
                                "parameter variables": obj["parameter variables"]}
                obj_with_prm = NodeObjects(new_obj).get_objects(new_obj["object"])
                obj_with_prm['6) line_no'] = obj["object"].lineno
                objects_with_prm.append(obj_with_prm)
                obj_code = ast.unparse(obj["object"])
        else:
            obj_with_prm = NodeObjects(obj).get_objects(obj["object"])
            objects_with_prm.append(obj_with_prm)

    return objects_with_prm


class NodeObjects:
    def __init__(self, obj):
        self.class_ = obj["class"]
        self.class_alias = obj["class alias"].split(".")[-1]
        self.obj = obj["object"]
        self.variable = ""
        self.parameter = []
        self.parameter_variables = obj["parameter variables"]
        self.variable_value = {}

    def get_objects(self, obj):
        self.get_values(obj)
        obj_dict = {"1) object": type(self.obj), "2) code": ast.unparse(self.obj), "3) class": self.class_,
                    "4) variable": self.variable, "5) parameter": self.parameter, "6) line_no": self.obj.lineno,
                    "7) value of parameter variables": self.variable_value}
        return obj_dict

    def get_values(self, obj):
        ast_type = type(obj)
        values = self.obj_type[ast_type](self, obj)
        return values

    def extract_func(self, obj):
        self.extract_args(obj.args)
        # decorator noch offen!
        self.get_values(obj.returns)
        return ast.unparse(obj)

    def extract_class(self, obj):
        for base in obj.bases:
            self.get_values(base)
        for keyword in obj.keywords:
            value = self.get_values(keyword.value)
            if self.class_alias in value:
                self.variable = keyword.arg
        return ast.unparse(obj)

    def extract_val(self, obj):
        self.get_values(obj.value)
        return ast.unparse(obj)

    def extract_delete(self, obj):
        for target in obj.targets:
            self.get_values(target)
        return ast.unparse(obj)

    def extract_assigns(self, obj):
        value = self.get_values(obj.value)
        if value is not None:  # for case of ann_assign
            if type(obj.value) == ast.Call:
                if ast.unparse(obj.value).startswith(self.class_alias):
                    self.variable = ast.unparse(obj.targets)
        return ast.unparse(obj)

    def extract_ann_assign(self, obj):
        value = self.get_values(obj.value)
        if value != None:
            self.variable = ast.unparse(obj.target)
        return ast.unparse(obj)

    def extract_for(self, obj):
        self.get_values(obj.iter)
        self.get_values(obj.target)
        return ast.unparse(obj)

    def extract_cond(self, obj):
        self.get_values(obj.test)
        return ast.unparse(obj)

    def extract_raise(self, obj):
        self.get_values(obj.cause)
        self.get_values(obj.exc)
        return ast.unparse(obj)

    def extract_with(self, obj):
        for item in obj.items:
            self.get_values(item.context_expr)
            self.get_values(item.optional_vars)
        return ast.unparse(obj)

    def extract_assert(self, obj):
        self.get_values(obj.msg)
        self.get_values(obj.test)
        return ast.unparse(obj)

    def extract_multiple_val(self, obj):
        for val in obj.values:
            self.get_values(val)
        return ast.unparse(obj)

    def extract_named_expr(self, obj):
        self.get_values(obj.target)
        self.get_values(obj.value)
        return ast.unparse(obj)

    def extract_bin_op(self, obj):
        self.get_values(obj.left)
        self.get_values(obj.right)
        return ast.unparse(obj)

    def extract_unary_op(self, obj):
        self.get_values(obj.operand)
        return ast.unparse(obj)

    def extract_lambda(self, obj):
        self.get_values(obj.body)
        return ast.unparse(obj)

    def extract_if_exp(self, obj):
        self.get_values(obj.body)
        self.get_values(obj.orelse)
        self.get_values(obj.test)
        return ast.unparse(obj)

    def extract_dict(self, obj):
        for key in obj.keys:
            self.get_values(key)
        for val in obj.values:
            self.get_values(val)
        return ast.unparse(obj)

    def extract_elements(self, obj):
        for ele in obj.elts:
            self.get_values(ele)
        return ast.unparse(obj)

    def extract_obj_comp(self, obj):
        self.get_values(obj.elt)
        for comp in obj.generators:
            self.extract_comprehension(comp)
        return ast.unparse(obj)

    def extract_compare(self, obj):
        for comp in obj.comparators:
            self.get_values(comp)
        self.get_values(obj.left)
        return ast.unparse(obj)

    def extract_call(self, obj):
        func = self.get_values(obj.func)
        if func == self.class_alias:
            for arg in obj.args:
                if type(arg) == ast.Name:
                    self.get_variable_scope(arg)
                argument = self.get_values(arg)
                self.parameter.append((None, argument))
            for param in obj.keywords:
                argument = str(param.arg)
                if type(param.value) == ast.Name:
                    self.get_variable_scope(param.value)
                value = self.get_values(param.value)
                self.parameter.append((argument, value))
        else:
            for arg in obj.args:
                self.get_values(arg)
            for param in obj.keywords:
                param_val = self.get_values(param.value)
                if param_val.startswith(self.class_alias):
                    self.variable = param.arg
        return ast.unparse(obj)

    def extracted_formatted_val(self, obj):
        self.get_values(obj.format_spec)
        self.get_values(obj.value)
        return ast.unparse(obj)

    def extract_unparse_obj(self, obj):
        return ast.unparse(obj)

    def extract_attr(self, obj):
        if obj.attr == self.class_alias:
            return obj.attr
        else:
            self.get_values(obj.value)

    def extract_subscript(self, obj):
        self.get_values(obj.slice)
        self.get_values(obj.value)
        return ast.unparse(obj)

    def extract_slice(self, obj):
        self.get_values(obj.lower)
        self.get_values(obj.upper)
        self.get_values(obj.step)
        return ast.unparse(obj)

    def extract_comprehension(self, obj):
        self.get_values(obj.iter)
        self.get_values(obj.target)
        for i in obj.ifs:
            self.get_values(i)
        return ast.unparse(obj)

    def extract_args(self, obj):
        args = []
        for arg in obj.posonlyargs:
            if arg.annotation == None:
                args.append(arg.arg)
        for arg in obj.args:
            if arg.annotation == None:
                args.append(arg.arg)
        if len(args) > 0:
            for default in obj.defaults:
                if default != None:
                    if self.class_alias in ast.unparse(default):
                        self.get_values(default)
                        self.variable = args[obj.defaults.index(default)]

        kw_only_args = []
        for arg in obj.kwonlyargs:
            if arg.annotation == None:
                kw_only_args.append(arg.arg)
        if len(kw_only_args) > 0:
            for kw in obj.kw_defaults:
                if kw != None:
                    if self.class_alias in ast.unparse(kw):
                        self.variable = kw_only_args[obj.kw_defaults.index(kw)]

        self.get_values(obj.kwarg)
        return ast.unparse(obj)

    def dummy(self, obj):
        if type(obj) != type(None):
            print(str(type(self.obj)) + ": dummy " + ast.unparse(self.obj))

    def get_variable_scope(self, obj):
        for assign in self.parameter_variables:
            for target in assign.targets:
                if ast.unparse(target) == ast.unparse(obj):
                    self.variable_value[ast.unparse(obj)] = ast.unparse(assign.value)

    obj_type = {ast.Name: extract_unparse_obj,
                ast.Constant: extract_unparse_obj,
                ast.Call: extract_call,
                ast.Assign: extract_assigns,
                ast.Expr: extract_val,
                ast.List: extract_elements,
                ast.Dict: extract_dict,
                ast.Attribute: extract_attr,
                ast.Tuple: extract_elements,
                ast.BinOp: extract_bin_op,
                ast.UnaryOp: extract_unary_op,
                ast.Assert: extract_assert,
                ast.Lambda: extract_lambda,
                ast.FunctionDef: extract_func,
                ast.For: extract_for,
                ast.Set: extract_elements,
                ast.Subscript: extract_subscript,
                ast.ClassDef: extract_class,
                ast.Return: extract_val,
                ast.With: extract_with,
                ast.Compare: extract_compare,
                ast.Slice: extract_slice,
                ast.BoolOp: extract_multiple_val,
                ast.ListComp: extract_obj_comp,
                ast.IfExp: extract_if_exp,
                ast.FormattedValue: extracted_formatted_val,
                ast.JoinedStr: extract_multiple_val,
                ast.While: extract_cond,
                ast.If: extract_cond,
                ast.AugAssign: extract_val,
                ast.AnnAssign: extract_assigns,
                ast.Starred: extract_val,
                ast.Delete: extract_delete,
                ast.Await: extract_val,
                ast.NamedExpr: extract_named_expr,
                ast.DictComp: extract_obj_comp,
                ast.SetComp: extract_obj_comp,
                ast.Raise: extract_raise,
                ast.AsyncFor: extract_for,
                ast.AsyncWith: extract_with,
                ast.Yield: extract_val,
                ast.YieldFrom: extract_val,
                ast.GeneratorExp: extract_obj_comp,
                ast.AsyncFunctionDef: extract_func,
                ast.Pass: dummy,
                ast.Break: dummy,
                ast.Continue: dummy,
                ast.Import: dummy,
                ast.ImportFrom: dummy,
                ast.Global: dummy,
                ast.Nonlocal: dummy,
                ast.Try: dummy,
                ast.excepthandler: dummy,
                type(None): dummy
                }


def merge_parameter(objects, classes):
    for obj in objects:
        cls_parameters = classes[obj["3) class"]]
        prj_parameters = obj['5) parameter']
        param_dict = {}
        asterisks = False
        for prj_param in prj_parameters:
            if prj_param[0] is None and not asterisks:
                for cls_param in list(cls_parameters):
                    if len(param_dict) <= list(cls_parameters).index(cls_param):
                        if cls_param != '*':
                            param_dict[cls_param] = prj_param[1]
                            break
                        else:
                            asterisks = True
                            break
            else:
                for cls_param in list(cls_parameters):
                    if cls_param == prj_param[0]:
                        param_dict[cls_param] = prj_param[1]
                        break
        obj['5) parameter'] = param_dict
    return objects


def get_variable_scope(objects):
    for obj in objects:
        scope = {}
        parameter = obj['5) parameter']
        for item in parameter.items():
            if item[1].startswith('ast.Name:'):
                item = list(item)
                item[1] = item[1].replace('ast.Name:', "")
                parameter[item[0]] = item[1]
                for assign in obj['7) parameter variables']:
                    for target in assign.targets:
                        if ast.unparse(target) == item[1]:
                            scope[item[1]] = ast.unparse(assign.value)
        obj["8) scope of parameter variables"] = scope


def convert_into_node_structure(project, objects):
    api_objects = []
    for obj in objects:
        dict_obj = {"project": project,
                    "class": obj["3) class"],
                    "line_no": obj["6) line_no"],
                    "variable": None,
                    "parameter": obj["5) parameter"],
                    "parameter_values": obj["7) value of parameter variables"]}
        if len(obj["4) variable"]) == 0:
            api_objects.append(dict_obj)
        else:
            for var in obj["4) variable"]:
                dict_obj["variable"] = var
                dict_copy = dict_obj.copy()
                api_objects.append(dict_copy)

    with open("node_objects.txt", 'w') as outfile:
        json.dump(api_objects, outfile, indent=4)


def main():
    project = "test_projects/another_test_project.py"
    class_objects_from_library = SklearnObjects(project).get_objects()
    classes = SklearnObjects(project).read_json()

    #project = "test_projects/torch_project.py"
    #class_objects_from_library = TorchObjects(project).get_objects()
    #classes = TorchObjects(project).read_json()

    objects_with_prm = get_parameters(class_objects_from_library)
    pprint(objects_with_prm)
    final_obj = merge_parameter(objects_with_prm, classes)
    # pprint(final_obj)
    # get_variable_scope(final_obj)
    #pprint(final_obj)
    convert_into_node_structure(project, final_obj)


if __name__ == "__main__":
    main()
