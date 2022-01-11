import ast

from obj_selector import CodeObjects
from pprint import pprint

#structur: .py-file
# class1 --- class2 --- class3
# (lineno --- variable - parameters)
def convert_into_data_structure(project, objects, classes):
    project = project
    class_name = objects[0]
    obj = objects[1]
    line_no = obj.lineno
    if type(obj) == ast.Assign:
        variable = obj.targets
    parameter =



def main():
    project = "test_projects/another_test_project.py"
    ml_lib = "sklearn"

    ast_objects = CodeObjects(ml_lib, project).get_objects()
    classes = CodeObjects(ml_lib, project).read_json()

    convert_into_data_structure(project, ast_objects[0], classes)
    pprint(ast_objects, width=250)


if __name__ == "__main__":
    main()