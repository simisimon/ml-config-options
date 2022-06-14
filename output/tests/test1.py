[
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
]