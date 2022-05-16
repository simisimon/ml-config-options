import ast


class ASTParameters:
    def __init__(self, obj):
        self.file = obj["file"]
        self.class_ = obj["class"]
        self.class_alias = obj["class alias"].split(".")[-1]
        self.obj = obj["object"]
        self.variable = ""
        self.parameter = []
        self.variable_value = {}

    def get_parameters(self, obj):
        self.get_value(obj)
        config_object = {"file": self.file, "class": self.class_, "parameter": self.parameter, "object": obj,
                    "parameter variables": self.variable_value, "variable": self.variable}
                    #"object": type(self.obj), "code": ast.unparse(self.obj), "line_no": self.obj.lineno
        return config_object

    def get_value(self, obj):
        ast_type = type(obj)
        value = self.obj_type[ast_type](self, obj)
        return value

    def extract_func(self, obj):
        self.extract_args(obj.args)
        # decorator noch offen!
        self.get_value(obj.returns)
        return ast.unparse(obj)

    def extract_class(self, obj):
        for base in obj.bases:
            self.get_value(base)
        for keyword in obj.keywords:
            value = self.get_value(keyword.value)
            if self.class_alias in value:
                self.variable = keyword.arg
        return ast.unparse(obj)

    def extract_val(self, obj):
        self.get_value(obj.value)
        return ast.unparse(obj)

    def extract_delete(self, obj):
        for target in obj.targets:
            self.get_value(target)
        return ast.unparse(obj)

    def extract_assigns(self, obj):
        value = self.get_value(obj.value)
        if value is not None:  # for case of ann_assign
            if type(obj.value) == ast.Call:
                if self.class_alias in ast.unparse(obj.value):
                    self.variable = ast.unparse(obj.targets)
        return ast.unparse(obj)

    def extract_ann_assign(self, obj):
        value = self.get_value(obj.value)
        if value is not None:
            self.variable = ast.unparse(obj.target)
        return ast.unparse(obj)

    def extract_for(self, obj):
        self.get_value(obj.iter)
        self.get_value(obj.target)
        return ast.unparse(obj)

    def extract_cond(self, obj):
        self.get_value(obj.test)
        return ast.unparse(obj)

    def extract_raise(self, obj):
        self.get_value(obj.cause)
        self.get_value(obj.exc)
        return ast.unparse(obj)

    def extract_with(self, obj):
        for item in obj.items:
            self.get_value(item.context_expr)
            self.get_value(item.optional_vars)
        return ast.unparse(obj)

    def extract_assert(self, obj):
        self.get_value(obj.msg)
        self.get_value(obj.test)
        return ast.unparse(obj)

    def extract_multiple_val(self, obj):
        for value in obj.values:
            self.get_value(value)
        return ast.unparse(obj)

    def extract_named_expr(self, obj):
        self.get_value(obj.target)
        self.get_value(obj.value)
        return ast.unparse(obj)

    def extract_bin_op(self, obj):
        self.get_value(obj.left)
        self.get_value(obj.right)
        return ast.unparse(obj)

    def extract_unary_op(self, obj):
        self.get_value(obj.operand)
        return ast.unparse(obj)

    def extract_lambda(self, obj):
        self.get_value(obj.body)
        return ast.unparse(obj)

    def extract_if_exp(self, obj):
        self.get_value(obj.body)
        self.get_value(obj.orelse)
        self.get_value(obj.test)
        return ast.unparse(obj)

    def extract_dict(self, obj):
        for key in obj.keys:
            self.get_value(key)
        for value in obj.values:
            self.get_value(value)
        return ast.unparse(obj)

    def extract_elements(self, obj):
        for ele in obj.elts:
            self.get_value(ele)
        return ast.unparse(obj)

    def extract_obj_comp(self, obj):
        self.get_value(obj.elt)
        for comp in obj.generators:
            self.extract_comprehension(comp)
        return ast.unparse(obj)

    def extract_compare(self, obj):
        for comp in obj.comparators:
            self.get_value(comp)
        self.get_value(obj.left)
        return ast.unparse(obj)

    def extract_call(self, obj):
        func = self.get_value(obj.func)
        if func == self.class_alias:
            for arg in obj.args:
                if type(arg) == ast.Name:
                    self.get_variable_scope(arg)
                argument = ast.unparse(arg)
                self.parameter.append((None, argument))
            for param in obj.keywords:
                argument = str(param.arg)
                if type(param.value) == ast.Name:
                    self.get_variable_scope(param.value)
                value = ast.unparse(param.value)
                self.parameter.append((argument, value))
        else:
            for arg in obj.args:
                self.get_value(arg)
            for param in obj.keywords:
                param_value = self.get_value(param.value)
                if self.class_alias in param_value:
                    self.variable = param.arg
        return ast.unparse(obj)

    def extracted_formatted_val(self, obj):
        self.get_value(obj.format_spec)
        self.get_value(obj.value)
        return ast.unparse(obj)

    def extract_unparse_obj(self, obj):
        return ast.unparse(obj)

    def extract_attr(self, obj):
        if obj.attr == self.class_alias:
            return obj.attr
        else:
            value = self.get_value(obj.value)
            return value

    def extract_subscript(self, obj):
        self.get_value(obj.slice)
        self.get_value(obj.value)
        return ast.unparse(obj)

    def extract_slice(self, obj):
        self.get_value(obj.lower)
        self.get_value(obj.upper)
        self.get_value(obj.step)
        return ast.unparse(obj)

    def extract_comprehension(self, obj):
        self.get_value(obj.iter)
        self.get_value(obj.target)
        for i in obj.ifs:
            self.get_value(i)
        return ast.unparse(obj)

    def extract_args(self, obj):
        args = []
        for arg in obj.posonlyargs:
            if arg.annotation is None:
                args.append(arg.arg)
        for arg in obj.args:
            if arg.annotation is None:
                args.append(arg.arg)
        if len(args) > 0:
            for default in obj.defaults:
                if default is not None:
                    if self.class_alias in ast.unparse(default):
                        self.get_value(default)
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

        self.get_value(obj.kwarg)
        return ast.unparse(obj)

    def none(self, obj):
        if type(obj) != type(None):
            print(str(type(self.obj)) + ": dummy " + ast.unparse(self.obj))

    def get_variable_scope(self, obj):
        self.variable_value[ast.unparse(obj)] = None

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
                ast.Pass: none,
                ast.Break: none,
                ast.Continue: none,
                ast.Import: none,
                ast.ImportFrom: none,
                ast.Global: none,
                ast.Nonlocal: none,
                ast.Try: none,
                ast.excepthandler: none,
                type(None): none
                }