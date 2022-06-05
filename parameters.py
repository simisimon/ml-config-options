import ast


class MLParameters:
    def __init__(self, ast_class_dict):
        self.file = ast_class_dict["file"]
        self.class_ = ast_class_dict["class"]
        self.class_alias = ast_class_dict["class alias"].split(".")[-1]
        self.obj = ast_class_dict["object"]
        self.scraped_parameters = ast_class_dict["scraped parameters"]["parameters"]
        self.variable = ""
        self.parameter = []
        self.variable_value = {}

    def get_parameters(self, ast_obj):
        self.get_value(ast_obj)
        self.merge_parameter()
        self.get_parameter_type()
        config_object = {"file": self.file, "class": self.class_, "object": ast_obj, "parameter": self.parameter,
                    "variable parameters": self.variable_value, "variable": self.variable}
        return config_object

    def get_value(self, obj):
        ast_type = type(obj)
        value = self.obj_type[ast_type](self, obj)
        return value

    def extract_func(self, obj):
        for decorator in obj.decorator_list:
            self.get_value(decorator)
        self.extract_args(obj.args)
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
                if type(arg) == ast.Name or type(arg) == ast.Attribute:
                    self.get_variable_scope(arg)
                argument = ast.unparse(arg)
                self.parameter.append((None, argument))
            for param in obj.keywords:
                argument = str(param.arg)
                if type(param.value) == ast.Name or type(param.value) == ast.Attribute:
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

        return ast.unparse(obj)

    def none(self, obj):
        pass

    def get_variable_scope(self, obj):
        self.variable_value[ast.unparse(obj)] = None

    def merge_parameter(self):
        parameter_dict = {}
        asterisks = False

        for code_prm in self.parameter:
            if code_prm[0] is None and not asterisks:
                for scraped_prm in list(self.scraped_parameters):
                    if scraped_prm[0] == "*" and len(scraped_prm) > 1:
                        if scraped_prm[1].isalpha():  # handling of *args
                            if scraped_prm not in parameter_dict:
                                if self.parameter.index(code_prm) == len(self.parameter) - 1:
                                    parameter_dict[scraped_prm] = code_prm[1]
                                else:
                                    if code_prm[1][0] == "*":
                                        if len(code_prm[1]) > 1:
                                            if code_prm[1][1] != "*":
                                                parameter_dict[scraped_prm] = code_prm[1]
                                    else:
                                        prm_tupel = "("
                                        for code_prm2 in self.parameter[self.parameter.index(code_prm):]:
                                            if code_prm2[1][0] != "*":
                                                prm_tupel += code_prm2[1] + ", "
                                        prm_tupel = prm_tupel[:-2] + ")"
                                        parameter_dict[scraped_prm] = prm_tupel
                            else:
                                if type(parameter_dict[scraped_prm]) == list:
                                     ls = parameter_dict[scraped_prm]
                                else:
                                    ls = []
                                    ls.append(parameter_dict[scraped_prm])
                                ls.append(code_prm[1])
                                parameter_dict[scraped_prm] = ls
                        break
                    elif len(parameter_dict) <= list(self.scraped_parameters).index(scraped_prm):
                        if scraped_prm == "*":
                            asterisks = True
                            break
                        else:
                            if code_prm[1][0] == '*' and len(code_prm[1]) > 1:
                                if code_prm[1][1] != '*':
                                    continue
                            else:
                                parameter_dict[scraped_prm] = code_prm[1]
                                break
            else:
                for scraped_prm in list(self.scraped_parameters):
                    if scraped_prm == code_prm[0]:
                        parameter_dict[scraped_prm] = code_prm[1]
                        break
                    elif scraped_prm[:2] == "**":  # handling of **kwargs
                        parameter_dict[scraped_prm] = code_prm[1]

        self.parameter = parameter_dict

    def get_parameter_type(self):
        for parameter in self.parameter:
            value = self.parameter[parameter]
            if type(value) == list:
                ast_type = 'List'
            else:
                ast_value = ast.parse(value).body[0].value
                ast_type = str(type(ast_value))
                ast_type = ast_type[12:-2]
            self.parameter[parameter] = {'value': value, 'type': ast_type}


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
                ast.AnnAssign: extract_ann_assign,
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


class InheritedParameters:
    def __init__(self, inherited_class):
        self.file = inherited_class["file"]
        self.class_ = inherited_class["class"]
        self.class_alias = inherited_class["class alias"].split(".")[-1]
        self.obj = inherited_class["object"]
        self.variable = ""
        self.parameter = {}
        self.variable_value = {}

    def get_parameters(self):
        for obj in self.obj.body:
            if type(obj) == ast.FunctionDef:
                if obj.name == '__init__':
                    for init_obj in obj.body:
                        if type(init_obj) == ast.Assign:
                            for target in init_obj.targets:
                                if type(target) == ast.Attribute:
                                    if ast.unparse(target.value) == "self":
                                        parameter = str(target.attr)
                                        ast_type = str(type(init_obj.value))
                                        ast_type = ast_type[12:-2]
                                        value = ast.unparse(init_obj.value)
                                        self.parameter[parameter]: {'value': value, 'type': ast_type}
                                        if type(init_obj.value) == ast.Name or type(init_obj.value) == ast.Attribute:
                                            self.variable_value[value] = None

        config_object = {"file": self.file, "class": self.class_, "object": self.obj, "parameter": self.parameter,
                    "variable parameters": self.variable_value, "variable": self.variable}
        return config_object


