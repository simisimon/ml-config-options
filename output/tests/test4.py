[
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
]