[
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
]