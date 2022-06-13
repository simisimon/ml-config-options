import ast
import json


class MLClasses:
    def __init__(self, file, library):
        self.library = library
        self.classes = {}
        self.file = file
        self.first_level_objects = []
        self.import_classes = {}
        self.import_other_objects = {}
        self.objects_from_library = []
        self.ast_classes = []

    def get_classes(self):
        self.read_json()
        self.get_first_level_objects()
        self.get_import_objects()
        self.get_objects_from_library()
        self.get_objects_containing_classes()
        return self.ast_classes

    def read_json(self):
        with open("output/scraped_classes/{0}.txt".format(self.library)) as json_file:
            self.classes = json.load(json_file)
        return self.classes

    def get_first_level_objects(self):
        with open(self.file, "r") as source:
            tree = ast.parse(source.read())
        self.first_level_objects = tree.body

    def get_import_objects(self):
        all_import_objects = {}
        for obj in self.first_level_objects:
            nodes = list(ast.walk(obj))
            for node in nodes:
                if type(node) == ast.Import:
                    for package in node.names:
                        if self.library in package.name:
                            if package.asname is not None:
                                all_import_objects[package.asname] = {"path": package.name}
                            else:
                                all_import_objects[package.name] = {"path": package.name}
                            if self.library not in all_import_objects:
                                all_import_objects[self.library] = {"path": self.library}
                elif type(node) == ast.ImportFrom:
                    for package in node.names:
                        module = node.module
                        if module is not None:
                            if self.library in module:
                                if package.asname is not None:
                                    all_import_objects[package.asname] = {"path": "{0}.{1}".format(module, package.name)}
                                else:
                                    all_import_objects[package.name] = {"path": "{0}.{1}".format(module, package.name)}

        for import_obj in all_import_objects:
            is_class = False
            for class_ in self.classes.items():
                if class_[1]["short name"] == import_obj:
                    is_class = True
                    break

            splitted_import_obj = import_obj.split(".")
            splitted_import_obj[0] = "id='{0}'".format(splitted_import_obj[0])
            for sio in enumerate(splitted_import_obj):
                if sio[0] > 0:
                    splitted_import_obj[sio[0]] = "attr='{0}'".format(sio[1])
            all_import_objects[import_obj].update({"ast style": splitted_import_obj})

            if is_class:
                if all_import_objects[import_obj]["path"] != class_[0]:
                    all_import_objects[import_obj].update({"path": class_[0]})
                self.import_classes[import_obj] = all_import_objects[import_obj]
            else:
                self.import_other_objects[import_obj] = all_import_objects[import_obj]

    def get_objects_from_library(self):
        for obj in self.first_level_objects:
            self.get_object(obj)

    def get_object(self, obj):
        if hasattr(obj, 'body'):
            for body_obj in obj.body:  # func, async func, class, with, async with, except handler, if...
                self.get_object(body_obj)
            obj.body = []

            if hasattr(obj, 'orelse'):  # control-flow-objects: for, if, async for, while
                for orelse_obj in obj.orelse:
                    self.get_object(orelse_obj)
                obj.orelse = []
                if hasattr(obj, 'handlers'):  # exception-objects: try
                    for handler_obj in obj.handlers:
                        self.get_object(handler_obj)
                    for finalbody_obj in obj.finalbody:
                        self.get_object(finalbody_obj)
                    obj.handlers = []
                    obj.finalbody = []

        if type(obj) != ast.Import and type(obj) != ast.ImportFrom:  # import-objects: import, import from
            dump_ast_obj = ast.dump(obj)
            import_values = dict(self.import_classes)
            import_values.update(self.import_other_objects)
            for import_obj in import_values.values():
                import_obj_findings = [i for i in import_obj["ast style"] if i in dump_ast_obj]
                if import_obj["ast style"] == import_obj_findings:
                    self.objects_from_library.append(obj)
                    break

    def get_objects_containing_classes(self):
        library_name_import = False
        for import_name, import_obj_values in self.import_other_objects.items():     # to detect classes that are not declared as the common path
            if import_obj_values['path'] == self.library:
                library_name_import = True
                library_alias = import_name

        for obj in self.objects_from_library:
            dump_ast_obj = ast.dump(obj)
            obj_code = ast.unparse(obj)
            for import_class_name, import_class_values in self.import_classes.items():
                class_occurence = 0
                for import_value in import_class_values["ast style"]:
                    class_string = "{0}(".format(import_class_name)
                    if class_string in obj_code:
                        indices = [i for i in range(len(obj_code)) if obj_code.startswith(class_string, i)]
                        for index in indices.copy():
                            if index == 0 or not(obj_code[index - 1].isalnum() or obj_code[index - 1] == "."):
                                class_occurence += 1
                                dump_ast_obj = dump_ast_obj.replace(import_value, "id='temp_class'", 1)
                                obj_code = obj_code.replace(class_string, "temp_class(", 1)
                            else:
                                indices.remove(index)
                        if class_occurence > 0:
                            obj_code = obj_code.replace("temp_class(", class_string)
                            lineno = obj.lineno
                            end_lineno = obj.end_lineno
                            try:
                                obj = ast.parse(obj_code).body[0]
                            except:
                                obj = ast.parse("{0}[]".format(obj_code)).body[0]
                            obj.lineno = lineno
                            obj.end_lineno = end_lineno
                            lib_class_obj = {"file": self.file, "class": import_class_values["path"],
                                             "class alias": class_string[:-1], "object": obj}
                            self.ast_classes.append(lib_class_obj)

            for import_name, import_obj_values in self.import_other_objects.items():
                import_obj_findings = [i for i in import_obj_values["ast style"] if i in dump_ast_obj]
                if import_obj_values["ast style"] == import_obj_findings:
                    for class_ in self.classes:
                        class_occurence = 0
                        if import_obj_values["path"] in class_:
                            class_string = "{0}(".format(class_.replace(import_obj_values["path"], import_name))
                            dump_class_name = "attr=\'{0}\',".format(class_.split(".")[-1])
                            if class_string in obj_code or dump_class_name in dump_ast_obj:
                                indices = [i for i in range(len(obj_code)) if obj_code.startswith(class_string, i)]
                                if len(indices) == 0:
                                    dump_ast = True
                                    indices = [i for i in range(len(obj_code)) if obj_code.startswith("{0}(".format(class_.split(".")[-1]), i)]
                                for index in indices:
                                    if index == 0 or not (obj_code[index - 1].isalnum() or obj_code[index - 1] == ".") or dump_ast:
                                        dump_ast = False
                                        class_occurence += 1
                                        dump_ast_obj = dump_ast_obj.replace("attr='{0}'".format(import_name.split(".")[-1]), "attr='temp_class'", 1)
                                        obj_code = obj_code.replace(class_string, "temp_class(", 1)
                                if class_occurence > 0:
                                    lib_class_obj = {"file": self.file, "class": class_, "class alias": class_string[:-1],
                                                     "object": obj}
                                    self.ast_classes.append(lib_class_obj)

class InheritedClasses(MLClasses):
    def __init__(self, file, library):
        MLClasses.__init__(self, file, library)

    def get_classes(self):
        self.read_json()
        self.get_first_level_objects()
        self.get_import_objects()
        self.get_inherited_classes()
        return self.ast_classes

    def get_inherited_classes(self):
        if len(self.import_classes) > 0 or len(self.import_other_objects) > 0:
            for first_level_obj in self.first_level_objects:
                for obj in list(ast.walk(first_level_obj)):
                    if type(obj) == ast.ClassDef:
                        for ast_base in obj.bases:
                            base = ast.unparse(ast_base)
                            for import_class in self.import_classes:
                                if import_class == base:
                                    class_path = self.import_classes[import_class]['path']
                                    config_obj = {"file": self.file, "class": class_path,
                                                  "class alias": import_class,
                                                  "object": obj,
                                                  "scraped parameters": None}
                                    self.ast_classes.append(config_obj)

                            for import_obj in self.import_other_objects:
                                if base.startswith("{0}.".format(import_obj)):
                                    class_path = self.import_other_objects[import_obj]['path']
                                    base = base.replace(import_obj, class_path, 1)
                                    if base in self.classes:
                                        config_obj = {"file": self.file, "class": base,
                                                      "class alias": base,
                                                      "object": obj,
                                                      "scraped parameters": None}
                                        self.ast_classes.append(config_obj)
                                    elif class_path in self.classes:
                                        self.import_other_objects[base]

