
# SODA Image Segmentation

This repository contains the code and resources necessary to replicate the image segmentation experiments for SourceData-NLP multimodal segmentation of compound figures. The project includes scripts for training and evaluating models, managing dependencies, and running the experiments in a Dockerized environment.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Experiment](#running-the-experiment)
  - [Evaluating Results](#evaluating-results)
- [Project Structure](#project-structure)
- [License](#license)

## Project Overview

The SODA Image Segmentation project aims to train a multimodal model to separate compound scientific figures into their constituent panels and match them to the correspondent panel captions. We followed a two-step procedure to achieve this goal. We first used object detection algorithms to separate the figure into its panels. Second, we used a multimodal LLM to extract the correspondent panel description from the figure caption, ensuring that the panel caption is understandable on its own, without the need of the context of the full figure caption. This repository provides all necessary scripts, notebooks, and configurations to replicate the experiments conducted in this project.

## Installation

### Prerequisites

- Python 3.9 or higher
- Docker (optional, but recommended for environment consistency)

### Steps

1. **Clone the repository:**

    \`\`\`bash
    git clone https://github.com/yourusername/soda_image_segmentation.git
    cd soda_image_segmentation
    \`\`\`

2. **Set up the environment:**

   You can set up the environment using \`virtualenv\` or \`conda\`, or you can use Docker.

   **Using \`virtualenv\` or \`conda\`:**

    \`\`\`bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use \`venv\Scripts\activate\`
    pip install -r requirements.txt
    \`\`\`

   **Using Docker:**

    \`\`\`bash
    docker-compose up --build
    \`\`\`

   This will build and start a Docker container with all dependencies installed.

   The docker container will initialized a JupyterLab server that can be used to interact with the code.

## Usage

### Running the Experiment

1. **Dataset:**
   
   The data needed for the dataset is contained in the folder `data/`. The original data can be found in the [EMBO HuggingFace Hub](https://huggingface.co/datasets/EMBO/SourceData).

    The data can also be generated running:

   python src/extract_figure_captions.py --input data/annotated_data.json --output data/figure_captions.jsonl

2. **Finetune the object detection model:**

    From the doker environment:

    \`\`\`bash
    python src/train_object_detection.py
    \`\`\`

    Evaluate the model performance on the SourceData dataset

    \`\`\`bash
    python src/evaluate_on_soda.py
    \`\`\`

3. **Match the extracted panels to their correspondent panel captions**

   \`\`\`bash
    python src/panel_label_matching.py
    \`\`\`

    Then check the results using the notebook provided on `notebooks/panel_matching_accuracy.ipynb`

## Project Structure

- \`src/\`: Contains the main source code for training and evaluating the model.
- \`notebooks/\`: Jupyter notebooks for analysis and evaluation.
- \`data/\`: Directory where datasets should be placed.
- \`runs/\`: Contains the outputs of the training runs, including model weights and evaluation metrics.
- \`Dockerfile\` and \`docker-compose.yml\`: Docker configurations for setting up the environment.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
