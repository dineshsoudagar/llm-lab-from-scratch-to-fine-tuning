import json


def load_json(json_file_path):
    """
    :param json_file_path: Path to json file
    """
    with open(json_file_path, "r") as file:
        data = json.load(file)
    print("Number of entries:", len(data))
    return data
