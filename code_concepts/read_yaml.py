import yaml

with open("../game_config.yaml", "r") as stream:
    print(yaml.safe_load(stream))