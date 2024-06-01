#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np

from PIL import Image

from lang_graph.Node import Node
from lang_graph.Graph import Graph
from lang_graph.utils.ArgParser import get_config

def create_graph(p, image_folder):

    image_dir = image_folder
    
    print(f"=========== Creating graph with images from {image_dir} ===========")

    graph = Graph()

    # Load all of the images and text descriptions
    for desp in p.node_descriptions:
        img = Image.open(f"{image_dir}/{desp[0]}")
        node = Node(img, desp[1], np.array(desp[2]))
        graph.add_node(node)
        print(f"---\nAdding node with:")
        print(f"    1. Original config {desp}")
        print(f"    2. Node {node}")

    print("=========== Done creating graph ===========")

    return graph

class ROSParamServer(rclpy.node.Node):

    def __init__(self):
        super().__init__("ros_param_server")

        self.declare_parameter("config_file", "config/config.yaml")
        self.declare_parameter("image_folder", "data/img")

    def get_params(self):
        return {
            "config_file": self.get_parameter("config_file").value,
            "image_folder": self.get_parameter("image_folder").value
        }

def main(args=None):

    # Setup the ROS2 node
    rclpy.init(args=args)

    # Get ROS2 parameters
    ros_param_server = ROSParamServer()

    # Get the configuration
    config_file = ros_param_server.get_params()["config_file"]
    image_folder = ros_param_server.get_params()["image_folder"]

    print(f"Configuration file: {config_file}")

    p = get_config(config_file)

    # Create the graph
    graph = create_graph(p, image_folder)

    # Encode the images
    graph.encode_images()

    # Encode the text
    graph.encode_text()

    # Get input text by the parser
    for text in p.input_text:
        node, _ = graph.query_text(text)

        # Log the node point
        print(f"Node point: {node}")

        print(f"=========== Querying graph with text: {text} ===========")

        # Display the most similar node
        node.display()


if __name__ == "__main__":
    main()