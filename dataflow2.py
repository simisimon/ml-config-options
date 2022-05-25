from scalpel.SSA.const import SSA
from scalpel.cfg import CFGBuilder
import ast

code_str = """
globalo = 0 

def global_def():
    globaloo = 0.0

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
            def func_c(self):
                c = 5
                b = 4444
"""

code_str = """

blu = 100

class cls:
    def __init__(self, b):
        a = 2
        x = b

load_model(b = 222)


def another_func(g):
    bla = 0
"""

class DataFlowAnalysis:
    def __init__(self, obj, variable):
        self.file = obj["file"]
        self.object = obj["object"]
        self.variable = 'b' #variable # "b"
        self.variable_value = []
        self.cfg_list = []
        self.parent_child = {}
        self.function_parameter_type = None

    def get_parameter_value(self):
        cfg = CFGBuilder().build_from_src(name="test", src=code_str)
        #cfg = CFGBuilder().build_from_file(name="test", filepath=self.file)
        self.cfg_list.append(("", cfg))
        self.get_cfgs(cfg)

        for cfg in self.cfg_list:
            self.get_SSA(cfg)

        self.variable_value = list(set(self.variable_value))
        print("file:    ", self.file)
        print("object:  ", ast.unparse(self.object))
        print("lineno:  ", self.object.lineno)
        print("variable:", self.variable)
        print("value:   ", self.variable_value)
        print(" ")
        return self.variable_value

    def get_cfgs(self, cfg):
        parent = cfg
        for class_cfg in cfg.class_cfgs.items():
            class_cfg = class_cfg[1]
            cfg_tuple = (parent, class_cfg)
            self.cfg_list.append(cfg_tuple)
            if parent in self.parent_child:
                self.parent_child[parent].append(class_cfg)
            else:
                self.parent_child[parent] = [class_cfg]
            self.get_cfgs(class_cfg)

        for function_cfg in cfg.functioncfgs.items():
            function_cfg = function_cfg[1]
            cfg_tuple = (parent, function_cfg)
            self.cfg_list.append(cfg_tuple)
            if parent in self.parent_child:
                self.parent_child[parent].append(function_cfg)
            else:
                self.parent_child[parent] = [function_cfg]
            self.get_cfgs(function_cfg)

    def get_SSA(self, cfg):
        m_ssa = SSA()
        _, const_dict = m_ssa.compute_SSA(cfg[1])
        for name, value in const_dict.items():
            if value is not None:
                if name[0] == self.variable:
                    self.variable_value.append(ast.unparse(value))

        for entryblock in cfg[1].entryblock.statements:
            if type(entryblock) == ast.FunctionDef:
                func_def_arg = self.check_function_definition(entryblock)
                if func_def_arg is not None:
                    func_def_arg["parent"] = cfg[1]
                    for functioncfg in cfg[1].functioncfgs.items():
                        if functioncfg[1].name == entryblock.name:
                            func_def_arg["cfg"] = functioncfg[1]
                            x = 2

        return None

    def check_function_definition(self, function):
        args = function.args
        func_def_arg = {}

        for arg in args.args:
            if arg.arg == self.variable:
                func_def_arg["type"] = "arg"
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
            if ast.unparse(args.vararg) == self.variable:
                func_def_arg["type"] = "vararg"
                return func_def_arg

        for kwonlyarg in args.kwonlyargs:
            if kwonlyarg.arg == self.variable:
                func_def_arg["type"] = "kwonlyarg"
                func_def_arg["keyword"] = kwonlyarg.arg
                index = args.kwonlyargs.index(kwonlyarg)
                kw_default = args.kw_defaults[index]
                if kw_default is not None:
                    self.variable_value.append(ast.unparse(kw_default))
                    return func_def_arg
                return func_def_arg

        for posonlyarg in args.posonlyargs:
            if posonlyarg.arg == self.variable:
                func_def_arg["type"] = "posonlyarg"
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
            if ast.unparse(args.kwarg) == self.variable:
                func_def_arg["type"] = "kwarg"
                return func_def_arg

        return None


