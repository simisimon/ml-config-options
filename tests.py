from main import SklearnOptions

def test1():
    file = 'output/tests/test1.py'
    repo = 'input_repo/tests'
    SklearnOptions(repo).get_config_options()
    expected_result = """[
    {
        "file": "tests/test1.py",
        "class": "sklearn.model_selection.ParameterSampler",
        "parameter": {
            "param_distributions": {
                "value": "self._params_space",
                "type": "Attribute"
            },
            "n_iter": {
                "value": "1",
                "type": "Constant"
            },
            "random_state": {
                "value": "self.random_state",
                "type": "Attribute"
            }
        },
        "variable parameters": {
            "self._params_space": {
                "0": {
                    "value": "self._get_params_space()",
                    "type": "Call"
                }
            },
            "self.random_state": {
                "0": {
                    "value": "np.random.RandomState(random_seed)",
                    "type": "Call"
                }
            }
        },
        "variable": "",
        "line no": 47
    }
]"""
    with open(file) as myfile:
        actual_result = myfile.read()
        assert expected_result == actual_result

def test2():
    file = 'output/tests/test2.py'
    repo = 'input_repo/tests'
    SklearnOptions(repo).get_config_options()
    expected_result = """[
    {
        "file": "tests/test2.py",
        "class": "torch.nn.Module",
        "parameter": {
            "x_scale": {
                "value": "x_scale",
                "type": "Name"
            },
            "y_scale": {
                "value": "y_scale",
                "type": "Name"
            }
        },
        "variable parameters": {
            "x_scale": {
                "0": {
                    "value": "next(iter(upsample_scales))",
                    "type": "Call"
                },
                "1": {
                    "value": "total_scale",
                    "type": "Name"
                },
                "2": {
                    "value": "scale",
                    "type": "Name"
                },
                "3": {
                    "value": "np.cumproduct(upsample_scales)[-1]",
                    "type": "Subscript"
                }
            },
            "y_scale": {
                "0": {
                    "value": "1",
                    "type": "Constant"
                }
            }
        },
        "variable": "",
        "line no": 47
    }
]"""
    with open(file) as myfile:
        actual_result = myfile.read()
        assert expected_result == actual_result

def test3():
    file = 'output/tests/test3.py'
    repo = 'input_repo/tests'
    SklearnOptions(repo).get_config_options()
    expected_result = """[
    {
        "file": "tests/test3.py",
        "class": "torch.utils.data.Dataset",
        "parameter": {
            "split": {
                "value": "split",
                "type": "Name"
            },
            "h": {
                "value": "h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')",
                "type": "Call"
            },
            "imgs": {
                "value": "self.h['images']",
                "type": "Subscript"
            },
            "cpi": {
                "value": "self.h.attrs['captions_per_image']",
                "type": "Subscript"
            },
            "transform": {
                "value": "transform",
                "type": "Name"
            },
            "dataset_size": {
                "value": "len(self.captions)",
                "type": "Call"
            }
        },
        "variable parameters": {
            "split": {},
            "transform": {
                "0": {
                    "value": "None",
                    "type": "Constant"
                }
            }
        },
        "variable": "",
        "line no": 8
    }
]"""
    with open(file) as myfile:
        actual_result = myfile.read()
        assert expected_result == actual_result

def test4():
    file = 'output/tests/test4.py'
    repo = 'input_repo/tests'
    SklearnOptions(repo).get_config_options()
    expected_result = """[
    {
        "file": "tests/test4.py",
        "class": "torch.torch.device",
        "parameter": {
            "type": {
                "value": "'cuda' if torch.cuda.is_available() else 'cpu'",
                "type": "IfExp"
            }
        },
        "variable parameters": {},
        "variable": "device",
        "line no": 16
    },
    {
        "file": "tests/test4.py",
        "class": "torch.utils.data.DataLoader",
        "parameter": {
            "dataset": {
                "value": "CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize]))",
                "type": "Call"
            },
            "batch_size": {
                "value": "1",
                "type": "Constant"
            },
            "shuffle": {
                "value": "True",
                "type": "Constant"
            },
            "num_workers": {
                "value": "1",
                "type": "Constant"
            },
            "pin_memory": {
                "value": "True",
                "type": "Constant"
            }
        },
        "variable parameters": {},
        "variable": "loader",
        "line no": 47
    }
]"""
    with open(file) as myfile:
        actual_result = myfile.read()
        assert expected_result == actual_result

def test5():
    file = 'output/tests/test5.py'
    repo = 'input_repo/tests'
    SklearnOptions(repo).get_config_options()
    expected_result = """[
    {
        "file": "tests/test5.py",
        "class": "torch.torch.device",
        "parameter": {
            "type": {
                "value": "'cuda' if torch.cuda.is_available() else 'cpu'",
                "type": "IfExp"
            }
        },
        "variable parameters": {},
        "variable": "device",
        "line no": 13
    }
]"""
    with open(file) as myfile:
        actual_result = myfile.read()
        assert expected_result == actual_result

def test6():
    file = 'output/tests/test6.py'
    repo = 'input_repo/tests'
    SklearnOptions(repo).get_config_options()
    expected_result = """[
        {
        "file": "tests/test6.py",
        "class": "sklearn.base.BaseEstimator",
        "parameter": {
            "n_clusters": {
                "value": "n_clusters",
                "type": "Name"
            },
            "metric": {
                "value": "metric",
                "type": "Name"
            },
            "method": {
                "value": "method",
                "type": "Name"
            },
            "init": {
                "value": "init",
                "type": "Name"
            },
            "max_iter": {
                "value": "max_iter",
                "type": "Name"
            },
            "tol": {
                "value": "tol",
                "type": "Name"
            },
            "random_state": {
                "value": "random_state",
                "type": "Name"
            }
        },
        "variable parameters": {
            "n_clusters": {
                "0": {
                    "value": "4",
                    "type": "Constant"
                },
                "1": {
                    "value": "self.n_clusters",
                    "type": "Attribute"
                }
            },
            "metric": {
                "0": {
                    "value": "'manhattan'",
                    "type": "Constant"
                }
            },
            "method": {
                "0": {
                    "value": "'per-axis'",
                    "type": "Constant"
                }
            },
            "init": {
                "0": {
                    "value": "'random'",
                    "type": "Constant"
                }
            },
            "max_iter": {
                "0": {
                    "value": "300",
                    "type": "Constant"
                }
            },
            "tol": {
                "0": {
                    "value": "0.0001",
                    "type": "Constant"
                }
            },
            "random_state": {
                "0": {
                    "value": "None",
                    "type": "Constant"
                }
            }
        },
        "variable": "",
        "line no": 20
    },
    {
        "file": "tests/test6.py",
        "class": "sklearn.base.ClusterMixin",
        "parameter": {
            "n_clusters": {
                "value": "n_clusters",
                "type": "Name"
            },
            "metric": {
                "value": "metric",
                "type": "Name"
            },
            "method": {
                "value": "method",
                "type": "Name"
            },
            "init": {
                "value": "init",
                "type": "Name"
            },
            "max_iter": {
                "value": "max_iter",
                "type": "Name"
            },
            "tol": {
                "value": "tol",
                "type": "Name"
            },
            "random_state": {
                "value": "random_state",
                "type": "Name"
            }
        },
        "variable parameters": {
            "n_clusters": {
                "0": {
                    "value": "4",
                    "type": "Constant"
                },
                "1": {
                    "value": "self.n_clusters",
                    "type": "Attribute"
                }
            },
            "metric": {
                "0": {
                    "value": "'manhattan'",
                    "type": "Constant"
                }
            },
            "method": {
                "0": {
                    "value": "'per-axis'",
                    "type": "Constant"
                }
            },
            "init": {
                "0": {
                    "value": "'random'",
                    "type": "Constant"
                }
            },
            "max_iter": {
                "0": {
                    "value": "300",
                    "type": "Constant"
                }
            },
            "tol": {
                "0": {
                    "value": "0.0001",
                    "type": "Constant"
                }
            },
            "random_state": {
                "0": {
                    "value": "None",
                    "type": "Constant"
                }
            }
        },
        "variable": "",
        "line no": 20
    }
]"""
    with open(file) as myfile:
        actual_result = myfile.read()
        assert expected_result == actual_result

def test7():
    file = 'output/tests/test7.py'
    repo = 'input_repo/tests'
    SklearnOptions(repo).get_config_options()
    expected_result = """[
    {
        "file": "tests/test7.py",
        "class": "tensorflow.io.gfile.GFile",
        "parameter": {
            "name": {
                "value": "path",
                "type": "Name"
            },
            "mode": {
                "value": "mode",
                "type": "Name"
            }
        },
        "variable parameters": {
            "path": {},
            "mode": {
                "0": {
                    "value": "'r'",
                    "type": "Constant"
                }
            }
        },
        "variable": "",
        "line no": 47
    }
]"""
    with open(file) as myfile:
        actual_result = myfile.read()
        assert expected_result == actual_result

def test8():
    file = 'output/tests/test8.py'
    repo = 'input_repo/tests'
    SklearnOptions(repo).get_config_options()
    expected_result = """[
    {
        "file": "tests/test8.py",
        "class": "sklearn.base.BaseEstimator",
        "parameter": {
            "_mode": {
                "value": "None",
                "type": "Constant"
            },
            "_ml_task": {
                "value": "None",
                "type": "Constant"
            },
            "_results_path": {
                "value": "None",
                "type": "Constant"
            },
            "_total_time_limit": {
                "value": "None",
                "type": "Constant"
            },
            "_model_time_limit": {
                "value": "None",
                "type": "Constant"
            },
            "_algorithms": {
                "value": "[]",
                "type": "List"
            },
            "_train_ensemble": {
                "value": "False",
                "type": "Constant"
            },
            "_stack_models": {
                "value": "False",
                "type": "Constant"
            },
            "_eval_metric": {
                "value": "None",
                "type": "Constant"
            },
            "_validation_strategy": {
                "value": "None",
                "type": "Constant"
            },
            "_verbose": {
                "value": "True",
                "type": "Constant"
            },
            "_explain_level": {
                "value": "None",
                "type": "Constant"
            },
            "_golden_features": {
                "value": "None",
                "type": "Constant"
            },
            "_features_selection": {
                "value": "None",
                "type": "Constant"
            },
            "_start_random_models": {
                "value": "None",
                "type": "Constant"
            },
            "_hill_climbing_steps": {
                "value": "None",
                "type": "Constant"
            },
            "_top_models_to_improve": {
                "value": "None",
                "type": "Constant"
            },
            "_random_state": {
                "value": "1234",
                "type": "Constant"
            },
            "_models": {
                "value": "[]",
                "type": "List"
            },
            "_best_model": {
                "value": "None",
                "type": "Constant"
            },
            "_threshold": {
                "value": "None",
                "type": "Constant"
            },
            "_metrics_details": {
                "value": "None",
                "type": "Constant"
            },
            "_max_metrics": {
                "value": "None",
                "type": "Constant"
            },
            "_confusion_matrix": {
                "value": "None",
                "type": "Constant"
            },
            "_data_info": {
                "value": "None",
                "type": "Constant"
            },
            "_model_subpaths": {
                "value": "[]",
                "type": "List"
            },
            "_stacked_models": {
                "value": "None",
                "type": "Constant"
            },
            "_fit_level": {
                "value": "None",
                "type": "Constant"
            },
            "_start_time": {
                "value": "time.time()",
                "type": "Call"
            },
            "_time_ctrl": {
                "value": "None",
                "type": "Constant"
            },
            "_all_params": {
                "value": "{}",
                "type": "Dict"
            },
            "n_features_in_": {
                "value": "None",
                "type": "Constant"
            },
            "tuner": {
                "value": "None",
                "type": "Constant"
            },
            "_boost_on_errors": {
                "value": "None",
                "type": "Constant"
            },
            "_kmeans_features": {
                "value": "None",
                "type": "Constant"
            },
            "_mix_encoding": {
                "value": "None",
                "type": "Constant"
            },
            "_max_single_prediction_time": {
                "value": "None",
                "type": "Constant"
            },
            "_optuna_time_budget": {
                "value": "None",
                "type": "Constant"
            },
            "_optuna_init_params": {
                "value": "{}",
                "type": "Dict"
            },
            "_optuna_verbose": {
                "value": "True",
                "type": "Constant"
            },
            "_n_jobs": {
                "value": "-1",
                "type": "UnaryOp"
            }
        },
        "variable parameters": {},
        "variable": "",
        "line no": 59
    }
]"""
    with open(file) as myfile:
        actual_result = myfile.read()
        assert expected_result == actual_result

def test9():
    file = 'output/tests/test9.py'
    repo = 'input_repo/tests'
    SklearnOptions(repo).get_config_options()
    expected_result = """[
    {
        "file": "tests/test9.py",
        "class": "sklearn.svm.LinearSVC",
        "parameter": {
            "loss": {
                "value": "loss",
                "type": "Name"
            },
            "random_state": {
                "value": "1",
                "type": "Constant"
            },
            "max_iter": {
                "value": "10",
                "type": "Constant"
            }
        },
        "variable parameters": {
            "loss": {}
        },
        "variable": "orig_est",
        "line no": 34
    }
]"""
    with open(file) as myfile:
        actual_result = myfile.read()
        assert expected_result == actual_result

def test10():
    file = 'output/tests/test10.py'
    repo = 'input_repo/tests'
    SklearnOptions(repo).get_config_options()
    expected_result = """[
    {
        "file": "tests/test10.py",
        "class": "sklearn.base.BaseEstimator",
        "parameter": {
            "a": {
                "value": "a",
                "type": "Name"
            },
            "b": {
                "value": "b",
                "type": "Name"
            }
        },
        "variable parameters": {
            "a": {
                "0": {
                    "value": "d",
                    "type": "Name"
                },
                "1": {
                    "value": "1",
                    "type": "Constant"
                },
                "2": {
                    "value": "e",
                    "type": "Name"
                }
            },
            "b": {
                "0": {
                    "value": "5",
                    "type": "Constant"
                }
            }
        },
        "variable": "",
        "line no": 3
    }
]"""
    with open(file) as myfile:
        actual_result = myfile.read()
        assert expected_result == actual_result
