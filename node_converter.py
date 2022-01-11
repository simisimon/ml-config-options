import ast

from obj_selector import CodeObjects
from pprint import pprint


def get_parameters(objects):
    for obj in objects:
        pprint(NodeObjects(obj).get_objects(obj[1]))


class NodeObjects:
    def __init__(self, obj):
        self.class_ = obj[0]
        self.obj = obj[1]
        self.variable = ""
        self.parameter = {}

    def get_objects(self, obj):
        self.get_values(obj)
        obj_dict = {"1) object": type(self.obj), "2) code": ast.unparse(self.obj), "3) class": self.class_,
                    "4) variable": self.variable, "5) parameter": self.parameter, "6) line_no": self.obj.lineno}
        return obj_dict

    def get_values(self, obj):
        ast_type = type(obj)
        values = self.obj_type[ast_type](self, obj)
        return values

    def extract_assigns(self, obj):
            #if type(obj.value) == ast.Call:
                #if obj.value.func.id == self.class_:
        self.variable = self.get_values(obj.targets[0])
        self.get_values(obj.value)

    def extract_expr(self, obj):
        self.get_values(obj.value)

    def extract_lambda(self, obj):
        return ast.unparse(obj)

    def extract_dict(self, obj):
        return ast.unparse(obj)

    def extract_call(self, obj):
        func = self.get_values(obj.func)
        if func == self.class_:
            for arg in obj.args:
                argument = self.get_values(arg)
                self.parameter[argument] = None
            for param in obj.keywords:
                argument = str(param.arg)
                value = self.get_values(param.value)
                self.parameter[argument] = value
        else:
            for arg in obj.args:
                self.get_values(arg)
            for param in obj.keywords:
                self.get_values(param.value)
            return ast.unparse(obj)

    def extract_constant(self, obj):
        return ast.unparse(obj)

    def extract_attr(self, obj):
        self.get_values(obj.value)
        return ast.unparse(obj)

    def extract_name(self, obj):
        id = obj.id
        return id

    def extract_list(self, obj):
        for ele in obj.elts:
            self.get_values(ele)
        return ast.unparse(obj)

    def extract_tuple(self, obj):
        for ele in obj.elts:
            self.get_values(ele)
        return ast.unparse(obj)

    def dummy(self, obj):
        print(str(type(self.obj)) + ": dummy " + ast.unparse(self.obj))

    obj_type = {ast.Name: extract_name,
                ast.Constant: extract_constant,
                ast.Call: extract_call,
                ast.Assign: extract_assigns,
                ast.Expr: extract_expr,
                ast.List: extract_list,
                ast.Dict: extract_dict,
                ast.Attribute: extract_attr,
                ast.Tuple: extract_tuple,
                ast.Assert: dummy,
                ast.Lambda: extract_lambda,
                ast.ClassDef: dummy,
                ast.With: dummy}


def convert_into_data_structure(project, objects, classes):
    # structure: .py-file
    # class1 --- class2 --- class3
    # (lineno --- variable - parameters)
    project = project
    class_name = objects[0]
    obj = objects[1]
    line_no = obj.lineno
    if type(obj) == ast.Assign:
        variable = obj.targets
    parameter = classes.get(class_name)
    parameter = ""


def main():
    project = "test_projects/another_test_project.py"
    ml_lib = "sklearn"

    ast_objects = CodeObjects(ml_lib, project).get_objects()
    classes = CodeObjects(ml_lib, project).read_json()

    get_parameters(ast_objects)
    #convert_into_data_structure(project, ast_objects[9], classes)
    pprint(ast_objects, width=250)


if __name__ == "__main__":
    main()


"""
obj_type = {ast.BoolOp: extract_bool_op,
            ,
            ,
            ast.FunctionDef: extract_func,
            ,
            ,
            ,
            ast.BinOp: extract_bin_op,
            ,
            ast.UnaryOp: extract_unary_op,
            ,
            ,
            ,
            ast.Subscript: extract_subscript,
            ast.Slice: extract_slice,
            ast.ExtSlice: extract_ext_slice,
            ast.Index: extract_index,
            ,
            ast.withitem: extract_withitem,
            ast.Import: dummy,
            ast.ImportFrom: dummy,
            ,
            ast.If: extract_if,
            ast.Return: extract_return,
            ---ast.ClassDef,
            ast.For: extract_for,
            ast.Compare: extract_compare,
            ast.JoinedStr: extract_joined_str,
            ast.FormattedValue: extract_formatted_val,
            ast.Pass: dummy,
            ast.Break: dummy,
            ast.Continue: dummy,
            ast.Delete: dummy,
            ,
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
"""