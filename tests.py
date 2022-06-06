from main import SklearnOptions

def test1():
    file = 'output/subject_systems/autogluon.txt'
    repo = 'input_repo/tests'
    SklearnOptions(repo).get_config_options()
    expected_result = """[
    {
        "file": "tests/test1.py",
        "class": "sklearn.decomposition.PCA",
        "parameter": {
            "n_components": "a"
        },
        "variable parameters": {
            "a": {
                "0": "1",
                "1": "5",
                "2": "e",
                "3": "f"
            }
        },
        "variable": "self.d",
        "line no": 5
    }
]"""
    with open(file) as myfile:
        actual_result = myfile.read()
        assert expected_result == actual_result