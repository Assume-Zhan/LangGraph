import argparse
import yaml # type: ignore

def _get_parser():

    parser = argparse.ArgumentParser(description="Training script for the model")

    # A list for text descriptions (a list of strings)
    parser.add_argument("--node_descriptions", type=list, default=["bed", "chair", "table"], help="List of text descriptions")

    # A list for input text
    parser.add_argument("--input_text", type=list, default=["Go to the sink with silver color"], help="List of input text")

    # Image directory
    parser.add_argument("--image_dir", type=str, default="./data/im", help="Directory containing images")
    
    # Add arguments for utility functions
    parser.add_argument("--config_folder", type=str, default="config", help="Folder containing configuration files")
    parser.add_argument("--config_file", type=str, default="config", help="Configuration file to be used")

    args = parser.parse_args()
    return parser, args.config_file, args.config_folder

def _load_yaml(filename, folder="config"):
    with open(f"{folder}/{filename}.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config

def get_config():

    # Get the parser and configuration file
    parser, config_file, config_folder = _get_parser()

    # Load the configuration file from folder
    config = _load_yaml(config_file, config_folder)

    # Set the default values to yaml configuration
    parser.set_defaults(**config)
    args = parser.parse_args(args = [])

    return args