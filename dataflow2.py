from scalpel.SSA.const import SSA
from scalpel.cfg import CFGBuilder
import ast

code_str = """
globalo = 0 

def global_def():
    globaloo = 0.0
    class D:
        f = 2

class A:
    def func_a(self):
        a = 1
        b = 1111
        
        def func_aa():
            aa = 2
            class AA:
                aaa = 3         
    class B:
        def func_b(self):
            b = 2222
            class_ = LinReg(b)
            b = 3333 

        class C:
            def __init__(self):
                x = 2
                
            def func_c(self):
                c = 5
                b = 4444
"""

code_str2 = """

blu = 100

#class cls:
 #   def __init__(self, b):
  #      a = 2
   #     x = b

def load_model(b = 11111):
    x = 2

def load_test(x = load_model(33333)):           ## yes
    x = 4

y = 5
x = [load_model(22222)]
x = y                                       ## yoah
y = 8


def another_func(x = 44444):
    x = 555555
    z = 666666 
    load_model(x)
    load_model(z)
    x = z
"""

code_str3 = """
def load_model(b = 0):
    b = 1
    b = x

c = 2
load_model(c)
load_model(3)

def func(d):
    e = 4
    load_model(d)
    load_model(e)

f = 5
func(f)
func(6)
"""

code_str2 = """
class C:
    def load_model(b = 0):
        b = 1
        b = x


load_model(3)
C.load_model(7)
"""

code_str = """
class HighwayNetwork(nn.Module):
    def __init__(self, size, /, g):
        super().__init__()
        self.W1 = nn.Linear(size, size)
        self.W2 = nn.Linear(size, size)
        self.W1.bias.data.fill_(0.)

channels = 2
hn = HighwayNetwork(channels)
"""


class DataFlowAnalysis:
    def __init__(self, obj, variable):
        self.file = obj["file"]
        self.object = obj["object"]
        self.variables = ['size'] #variable # "b"
        self.variable_value = []
        self.cfg_list = []
        self.possible_object_paths = {}
        self.function_parameter_type = None
        self.all_func_calls = []

    def get_parameter_value(self):
        cfg = CFGBuilder().build_from_src(name="test", src=code_str)
        #cfg = CFGBuilder().build_from_file(name="test", filepath=self.file)
        self.cfg_list.append(cfg)
        self.get_cfgs(cfg)
        self.get_all_func_calls()
        self.get_all_possible_object_paths()

        for variable in self.variables:
            func_def_args = []
            for cfg in self.cfg_list:
                self.get_SSA(cfg, variable)

            for cfg in self.cfg_list:
                for entryblock in cfg.entryblock.statements:
                    if type(entryblock) == ast.FunctionDef:
                        func_def_arg = self.check_function_definition(entryblock, variable)
                        if func_def_arg is not None:
                            for func_cfg in cfg.functioncfgs.keys():
                                if func_cfg[1] == entryblock.name:
                                    func_def_arg["function paths"] = self.possible_object_paths[cfg.functioncfgs[func_cfg]]
                                    break

                            func_def_arg["variable"] = variable
                            func_def_arg["parent"] = cfg
                            func_def_args.append(func_def_arg)

            for cfg in self.cfg_list:
                for func_def_arg in func_def_args:
                    self.get_func_calls(func_def_arg, cfg)

        self.variable_value = list(set(self.variable_value))
        print("file:    ", self.file)
        print("object:  ", ast.unparse(self.object))
        print("lineno:  ", self.object.lineno)
        print("variable:", self.variables)
        print("value:   ", self.variable_value)
        print(" ")
        return self.variable_value

    def get_cfgs(self, cfg):
        parent = cfg
        for class_cfg in cfg.class_cfgs.items():
            class_cfg = class_cfg[1]
            self.cfg_list.append(class_cfg)
            self.possible_object_paths[class_cfg] = parent
            self.get_cfgs(class_cfg)

        for function_cfg in cfg.functioncfgs.items():
            function_cfg = function_cfg[1]
            self.cfg_list.append(function_cfg)
            self.possible_object_paths[function_cfg] = parent
            self.get_cfgs(function_cfg)

    def get_all_func_calls(self):
        for cfg in self.cfg_list:
            for statement in cfg.entryblock.statements:
                nodes = list(ast.walk(statement))
                for node in nodes:
                    if type(node) == ast.Call:
                        self.all_func_calls.append(node)
                    if type(node) == ast.FunctionDef or type(node) == ast.AsyncFunctionDef:
                        for default in node.args.defaults:
                            nodes_prm = list(ast.walk(default))
                            for node_prm in nodes_prm:
                                if type(node_prm) == ast.Call:
                                    self.all_func_calls.append(node_prm)
                        for default in node.args.kw_defaults:
                            nodes_prm = list(ast.walk(default))
                            for node_prm in nodes_prm:
                                if type(node_prm) == ast.Call:
                                    self.all_func_calls.append(node_prm)

    def get_all_possible_object_paths(self):
        parent_child_relation = {}

        for child in self.possible_object_paths.keys():
            parents = self.parent_child_match(child)
            init = False

            if child.name == "__init__":
                init = True
                parent_child_relation[child] = [parents[0].name]
            else:
                parent_child_relation[child] = [child.name]
            prev_parent = None
            for parent in parents:
                if init:
                    parent_child_relation[child] = [parents[0].name]
                    init = False
                    continue

                if parent == self.cfg_list[0]:
                    continue

                for func_parent in parent.functioncfgs.items():
                    if prev_parent == func_parent[1]:
                        parent_child_relation[child].remove(parent_child_relation[child][-1])
                        #parent_child_relation[child].remove(prev_parent)

                path = "{0}.{1}".format(parent.name, parent_child_relation[child][-1])
                parent_child_relation[child].append(path)

                prev_parent = parent

        self.possible_object_paths = parent_child_relation

    def parent_child_match(self, child):
        parent_child_relation = []
        parent = self.possible_object_paths[child]
        parent_child_relation.append(parent)
        for child in self.possible_object_paths.keys():
            if child == parent:
                parent_child_relation.extend(self.parent_child_match(child))
                break
        return parent_child_relation

    def get_SSA(self, cfg, variable):
        m_ssa = SSA()
        _, const_dict = m_ssa.compute_SSA(cfg)
        for name, value in const_dict.items():
            if value is not None:
                if name[0] == variable:
                    if type(value) == ast.Name:
                        if ast.unparse(value) not in self.variables:
                            self.variables.append(ast.unparse(value))
                    else:
                        self.variable_value.append(ast.unparse(value))

    def check_function_definition(self, function, variable):
        args = function.args
        func_def_arg = {}

        for arg in args.args:
            if arg.arg == variable:
                func_def_arg["type"] = "arg"
                if ast.unparse(args.args[0]) == "self":
                    func_def_arg["index"] = args.args.index(arg) - 1
                else:
                    func_def_arg["index"] = args.args.index(arg)

                col_offset = arg.col_offset
                for default in args.defaults:
                    if default.col_offset > col_offset:
                        index = args.args.index(arg)
                        length = len(args.args) - 1
                        if index < length:
                            col_offset_next = args.args[index + 1].col_offset
                            if default.col_offset < col_offset_next:
                                self.variable_value.append(ast.unparse(default))
                                return func_def_arg
                        else:
                            self.variable_value.append(ast.unparse(default))
                            return func_def_arg
                return func_def_arg

        if args.vararg is not None:
            if ast.unparse(args.vararg) == variable:
                func_def_arg["type"] = "vararg"
                return func_def_arg

        for kwonlyarg in args.kwonlyargs:
            if kwonlyarg.arg == variable:
                func_def_arg["type"] = "kwonlyarg"
                func_def_arg["keyword"] = kwonlyarg.arg
                index = args.kwonlyargs.index(kwonlyarg)
                kw_default = args.kw_defaults[index]
                if kw_default is not None:
                    self.variable_value.append(ast.unparse(kw_default))
                    return func_def_arg
                return func_def_arg

        for posonlyarg in args.posonlyargs:
            if posonlyarg.arg == variable:
                func_def_arg["type"] = "posonlyarg"
                if ast.unparse(args.posonlyargs[0]) == "self":
                    func_def_arg["index"] = args.posonlyargs.index(posonlyarg) - 1
                else:
                    func_def_arg["index"] = args.posonlyargs.index(posonlyarg)
                col_offset = posonlyarg.col_offset
                for default in args.defaults:
                    if default.col_offset > col_offset:
                        index = args.posonlyargs.index(posonlyarg)
                        length = len(args.posonlyargs) - 1
                        if index < length:
                            col_offset_next = args.posonlyargs[index + 1].col_offset
                            if default.col_offset < col_offset_next:
                                self.variable_value.append(ast.unparse(default))
                                return func_def_arg
                        else:
                            self.variable_value.append(ast.unparse(default))
                            return func_def_arg
                return func_def_arg

        if args.kwarg is not None:
            if ast.unparse(args.kwarg) == variable:
                func_def_arg["type"] = "kwarg"
                return func_def_arg

        return None

    def get_func_calls(self, func_def_arg, cfg):
        variable_values = []
        entryblock = cfg.entryblock
        for func_call in entryblock.func_calls:
            if func_call["name"] in func_def_arg["function paths"]:   ## self?
                for statement in entryblock.statements:
                    nodes = list(ast.walk(statement))
                    for node in nodes:
                        if type(node) == ast.Call:
                            if ast.unparse(node.func) in func_def_arg["function paths"]: #or ast.unparse(node.func) == "self.{0}".format(func_def_arg["func"]):
                                if func_def_arg["type"] == "arg":
                                    found_keyword = False
                                    for keyword in node.keywords:
                                        if keyword.arg == func_def_arg["variable"]:
                                            found_keyword = True
                                            variable_values.append(keyword.value)
                                            break
                                    if not found_keyword:
                                        variable_values.append((node.args[func_def_arg["index"]]))

        for variable_value in variable_values:
            if type(variable_value) == ast.Name: ## self.?
                if ast.unparse(variable_value) not in self.variables:
                    self.variables.append(ast.unparse(variable_value))
            else:
                self.variable_value.append(ast.unparse(variable_value))