import json
from iteration_utilities import unique_everseen

def get_numbers(file_name: str):
    class_count = 0
    class_option_count = 0
    method_count = 0
    method_option_count = 0


    with open(file_name, "r", encoding="utf-8") as src:
        data = json.load(src)

        for item in data:
            if item["name"][0].isupper():
                class_count += 1
                class_option_count += len(item["params"])
            
            if item["name"][0].islower():
                method_count += 1
                method_option_count += len(item["params"])

        
    print("==========================")
    print("Class Count: ", class_count)
    print("Class Option Count: ", class_option_count)
    print("Method Count: ", method_count)
    print("Method Option Count: ", method_option_count)

def main():
    get_numbers("sklearn_default_values.json")
    get_numbers("tensorflow_default_values.json")
    get_numbers("torch_default_values.json")


if __name__ == "__main__":
    main()
