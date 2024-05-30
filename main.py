import numpy as np
import torch
import clip

from PIL import Image

from Node import Node
from Graph import Graph
from utils.ArgParser import get_config

def create_graph(p):

    image_dir = p.image_dir
    
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

def main():
    p = get_config()

    # Create the graph
    graph = create_graph(p)

    # Encode the images
    graph.encode_images()

    # Encode the text
    graph.encode_text()

    # Get input text by the parser
    for text in p.input_text:
        node, _ = graph.query_text(text)

        print(f"=========== Querying graph with text: {text} ===========")

        # Display the most similar node
        node.display()


if __name__ == "__main__":
    main()