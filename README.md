# LangGraph

Use CLIP for creating the graph with image and language embeddings

## Installation

### Prepare data

Download the testing image sceenshot from the gazebo world on the google drive [link](https://drive.google.com/drive/u/1/folders/1JCm88xQY8XugXqoJuo6_jMQ8Cx_p8gx_).
Please put the images in the folder `LangGraph/data/img/`, and the structure will be like above.

```
LangGraph
├── config
├── data
|   └── img
|       ├── .gitkeep
|       ├── barbell.png
|       ├── bed.png
|       ├── blue_chair.png
|       └── ...
├── docker
|   ├── compose.yaml
|   └── Dockerfile
├── utils
├── .gitignore
└── README.md
```

### Build with Docker

- Go to the docker folder
    ```bash
    # Assume you are in the root folder of the project
    cd docker
    ```
- Build the docker image with compose file
    ```bash
    docker compose run --rm --build clip-pytorch
    ```
## Run

### Test with image screenshot from the gazebo world

- Run the container with the compose file
    ```bash
    docker compose run --rm clip-pytorch-infer
    ```

### Add node to graph

First, you need to add a new image to the folder `LangGraph/data/img/`, and update the argument in config file `LangGraph/config/config.yaml`, such as above.

```yaml
node_descriptions: [
    ["yellow_table_with_ipad.png", "Go to the yellow table with ipad", [0., 0.]],
    ["yellow_table_with_tomato_table.png", "Go to the yellow table with tomato picture", [0., 0.]],
]
```

### Query the graph

Change the argument in the config file `LangGraph/config/config.yaml` to the image you want to query, such as above.

```yaml
input_text: [
  "Go to the chairs with blue color"
]
```