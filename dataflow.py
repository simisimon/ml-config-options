import ast


class DataFlowAnalysis:
    def __init__(self, obj, variable):
        self.file = obj["file"]
        self.object_dict = {"function": None, "objects": [], "variable": variable,
                            "line_no": obj["object"].lineno, "last_assign_line_no": 0}
        self.variable_value = {}
        self.counter = 0
        self.function_parameter = None

    def get_parameter_value(self):
        self.get_parent_func()
        self.remove_objects(self.object_dict)
        self.get_last_assignment(self.object_dict)
        if not self.variable_value:
            self.check_function_parameter()
        self.detect_deeper_objects(self.object_dict)
        if self.function_parameter is not None:
            function_calls = self.detect_function_calls()
            for object_dict in function_calls:
                self.remove_objects(object_dict)
                self.get_last_assignment(object_dict)
                self.detect_deeper_objects(object_dict)
        return self.variable_value

    def get_parent_func(self):
        with open(self.file, "r") as source:
            tree = ast.parse(source.read())

        first_level_objects = tree.body
        for obj in first_level_objects:
            if obj.end_lineno < self.object_dict["line_no"]:
                continue
            else:
                nodes = list(ast.walk(obj))
                nodes.insert(0, obj)
                for node in nodes:
                    if obj.lineno > self.object_dict["line_no"]:
                        break
                    if type(node) == ast.FunctionDef or type(node) == ast.AsyncFunctionDef:
                        if node.lineno <= self.object_dict["line_no"]:
                            if self.object_dict["line_no"] <= node.end_lineno:
                                self.object_dict["function"] = node
                                self.object_dict["objects"] = node.body
                                break
                        else:
                            break
                if self.object_dict["function"] is not None:
                    break

    def remove_objects(self, object_dict):
        for obj in object_dict["objects"]:
            if obj.lineno > object_dict["line_no"]:
                object_dict["objects"].remove(obj)

    def get_last_assignment(self, object_dict):
        for obj in object_dict["objects"]:
            if type(obj) == ast.Assign:
                for target in obj.targets:
                    if ast.unparse(target) == object_dict["variable"]:
                        self.variable_value[self.counter] = ast.unparse(obj.value)
                        object_dict["last_assign_line_no"] = obj.lineno
                        self.counter += 1

    def check_function_parameter(self):
        if self.object_dict["function"] is not None:
            args = self.object_dict["function"].args

            for arg in args.args:
                if arg.arg == self.object_dict["variable"]:
                    self.function_parameter = "arg"
                    col_offset = arg.col_offset
                    for default in args.defaults:
                        if default.col_offset > col_offset:
                            index = args.args.index(arg)
                            length = len(args.args) - 1
                            if index < length:
                                col_offset_next = args.args[index + 1].col_offset
                                if default.col_offset < col_offset_next:
                                    self.variable_value[self.counter] = ast.unparse(default)
                                    self.counter += 1
                            else:
                                self.variable_value[self.counter] = ast.unparse(default)
                                self.counter += 1
                            return
                    return

            if args.vararg is not None:
                if ast.unparse(args.vararg) == self.object_dict["variable"]:
                    self.function_parameter = "vararg"
                    return

            for kwonlyarg in args.kwonlyargs:
                if kwonlyarg.arg == self.object_dict["variable"]:
                    self.function_parameter = "kwonlyarg"
                    index = args.kwonlyargs.index(kwonlyarg)
                    kw_default = args.kw_defaults[index]
                    if kw_default is not None:
                        self.variable_value[self.counter] = ast.unparse(kw_default)
                        self.counter += 1
                        return
                    return

            for posonlyarg in args.posonlyargs:
                if posonlyarg.arg == self.object_dict["variable"]:
                    self.function_parameter = "posonlyarg"
                    col_offset = posonlyarg.col_offset
                    for default in args.defaults:
                        if default.col_offset > col_offset:
                            index = args.posonlyargs.index(posonlyarg)
                            length = len(args.posonlyargs) - 1
                            if index < length:
                                col_offset_next = args.posonlyargs[index + 1].col_offset
                                if default.col_offset < col_offset_next:
                                    self.variable_value[self.counter] = ast.unparse(default)
                                    self.counter += 1
                                    return
                            else:
                                self.variable_value[self.counter] = ast.unparse(default)
                                self.counter += 1
                                return
                    return

            if args.kwarg is not None:
                if ast.unparse(args.kwarg) == self.object_dict["variable"]:
                    self.function_parameter = "kwarg"
                    return

    def detect_deeper_objects(self, object_dict):
        for obj in object_dict["objects"]:
            if obj.lineno > object_dict["last_assign_line_no"]:
                self.get_objects(obj, object_dict["variable"])

    def get_objects(self, obj, variable):
        if hasattr(obj, 'body'):
            for body_obj in obj.body:  # func, async func, class, with, async with, except handler, if...
                self.get_objects(body_obj, variable)
            if hasattr(obj, 'orelse'):  # control-flow-objects: for, if, async for, while
                for orelse_obj in obj.orelse:
                    self.get_objects(orelse_obj, variable)

                if hasattr(obj, 'handlers'):  # exception-objects: try
                    for handler_obj in obj.handlers:
                        self.get_objects(handler_obj, variable)
                    for finalbody_obj in obj.finalbody:
                        self.get_objects(finalbody_obj, variable)

        if type(obj) == ast.Assign:
            for target in obj.targets:
                if ast.unparse(target) == variable:
                    self.variable_value[self.counter] = ast.unparse(obj.value)
                    self.counter += 1

    def detect_function_calls(self):
        function_calls = []
        global_assignments = []

        with open(self.file, "r") as source:
            tree = ast.parse(source.read())

        first_level_objects = tree.body
        for obj in first_level_objects:
            if type(obj) == ast.Assign:
                global_assignments.append(obj)
            nodes = list(ast.walk(obj))
            for node in nodes:
                if type(node) == ast.Call:
                    if ast.unparse(node.func) == self.object_dict["function"].name or ast.unparse(node.func) == "self.{0}".format(self.object_dict["function"].name):
                        if self.function_parameter == "arg":
                            found_keyword = False
                            for keyword in node.keywords:
                                if keyword.arg == self.object_dict["variable"]:
                                    found_keyword = True
                                    if type(keyword.value) == ast.Name:
                                        if hasattr(obj, 'body'):
                                            objects = obj.body
                                        else:
                                            objects = global_assignments
                                        dict_obj = {"objects": objects, "variable": keyword.value.id,
                                                    "line_no": node.lineno, "last_assign_line_no": 0}
                                        function_calls.append(dict_obj)
                                    else:
                                        self.variable_value[self.counter] = ast.unparse(keyword.value)
                                        self.counter += 1
                                    break
                            if not found_keyword:
                                args = self.object_dict["function"].args
                                col_offset_list = []
                                for arg in args.args:
                                    if arg.arg == self.object_dict["variable"]:
                                        col_offset = arg.col_offset
                                    if arg.arg != "self":
                                        col_offset_list.append(arg.col_offset)
                                if args.kwarg is not None:
                                    col_offset_list.append(args.kwarg.col_offset)
                                for kwonlyarg in args.kwonlyargs:
                                    col_offset_list.append(kwonlyarg.col_offset)
                                for posonlyarg in args.posonlyargs:
                                    if posonlyarg.arg != "self":
                                        col_offset_list.append(posonlyarg.col_offset)
                                if args.vararg is not None:
                                    col_offset_list.append(args.vararg.col_offset)
                                col_offset_list.sort()
                                index = col_offset_list.index(col_offset)
                                variable_ast = node.args[index]
                                if type(variable_ast) == ast.Name:
                                    if hasattr(obj, 'body'):
                                        objects = obj.body
                                    else:
                                        objects = global_assignments
                                    dict_obj = {"objects": objects, "variable": variable_ast.id,
                                                "line_no": node.lineno, "last_assign_line_no": 0}
                                    function_calls.append(dict_obj)
                                else:
                                    self.variable_value[self.counter] = ast.unparse(variable_ast)
                                    self.counter += 1

                        elif self.function_parameter == "kwonlyarg":
                            for keyword in node.keywords:
                                if keyword.arg == self.object_dict["variable"]:
                                    if type(keyword.value) == ast.Name:
                                        if hasattr(obj, 'body'):
                                            objects = obj.body
                                        else:
                                            objects = global_assignments
                                        dict_obj = {"objects": objects, "variable": keyword.value.id,
                                                    "line_no": node.lineno, "last_assign_line_no": 0}
                                        function_calls.append(dict_obj)
                                    else:
                                        self.variable_value[self.counter] = ast.unparse(keyword.value)
                                        self.counter += 1

                        elif self.function_parameter == "vararg":
                            pre_parameter_length = len(self.object_dict["function"].args.args) + len(self.object_dict["function"].args.posonlyargs)
                            parameter_tuples = ()
                            for arg in node.args[pre_parameter_length:]:
                                parameter_tuples += (ast.unparse(arg),)
                            if len(parameter_tuples) == 1:
                                if type(arg) == ast.Name:
                                    if hasattr(obj, 'body'):
                                        objects = obj.body
                                    else:
                                        objects = global_assignments
                                    dict_obj = {"objects": objects, "variable": ast.unparse(arg),
                                                "line_no": node.lineno, "last_assign_line_no": 0}
                                    function_calls.append(dict_obj)
                                else:
                                    self.variable_value[self.counter] = ast.unparse(arg)
                                    self.counter += 1
                            else:
                                self.variable_value[self.counter] = parameter_tuples
                                self.counter += 1

                        elif self.function_parameter == "kwarg":
                            if len(node.keywords) > 0:
                                keyword = node.keywords[-1]
                                kwonlyargs = [arg.arg for arg in self.object_dict["function"].args.kwonlyargs]
                                if keyword.arg not in kwonlyargs:
                                    self.counter += 1
                                    if keyword.arg is None:
                                        self.variable_value[self.counter] = ast.unparse(keyword.value)
                                    else:
                                        kwarg_dict = {keyword.arg: ast.unparse(keyword.value)}
                                        self.variable_value[self.counter] = kwarg_dict

                        elif self.function_parameter == "posonlyarg":
                            for posonlyarg in self.object_dict["function"].args.posonlyargs:
                                if posonlyarg.arg == self.object_dict["variable"]:
                                    index = self.object_dict["function"].args.posonlyargs.index(posonlyarg)
                                    break
                            if len(node.args) > 0:
                                posonlyarg = node.args[index]
                                if type(posonlyarg) == ast.Name:
                                    if hasattr(obj, 'body'):
                                        objects = obj.body
                                    else:
                                        objects = global_assignments
                                    dict_obj = {"objects": objects, "variable": posonlyarg.id,
                                                "line_no": node.lineno, "last_assign_line_no": 0}
                                    function_calls.append(dict_obj)
                                else:
                                    self.variable_value[self.counter] = ast.unparse(posonlyarg)
                                    self.counter += 1

        return function_calls