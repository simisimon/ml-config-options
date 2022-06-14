[
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
]