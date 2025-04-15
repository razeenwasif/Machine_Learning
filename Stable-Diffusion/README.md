# Stable Diffusion from Scratch in PyTorch

## Overview

This project is an implementation of the Stable Diffusion text-to-image model from scratch using PyTorch. The goal is to provide a clear and understandable codebase for educational purposes, demonstrating the core components and mechanisms behind latent diffusion models.

This implementation focuses on building the essential parts of the architecture, including:

* **Variational Autoencoder (VAE):** For encoding images into a latent space and decoding latents back into images.
* **U-Net:** The core noise prediction network that operates in the latent space.
* **Text Encoder:** Integration with a pre-trained text encoder (e.g., CLIP's text transformer) to condition the diffusion process on text prompts.
* **Noise Scheduler:** Implementation of a diffusion scheduler (e.g., DDPM, DDIM) to manage the noise addition and removal process during training and inference.

**Disclaimer:** This is an educational implementation built from scratch. While aiming for correctness, it may not be as optimized or feature-rich as official implementations like those found in libraries such as Hugging Face's `diffusers`.

## Architecture

The model follows the standard latent diffusion architecture:

1.  **Text Encoding:** A text prompt is converted into numerical embeddings using a pre-trained text encoder (e.g., CLIP).
2.  **Image Encoding (Training):** Input images are encoded into a lower-dimensional latent space using the VAE encoder.
3.  **Diffusion Process (Training):** Noise is progressively added to the latent representation according to a defined schedule. The U-Net is trained to predict the noise added at each step, conditioned on the text embeddings.
4.  **Denoising Process (Inference):** Starting with random noise in the latent space, the trained U-Net iteratively removes noise, guided by the text prompt's embeddings and the noise schedule, to produce a clean latent representation.
5.  **Image Decoding (Inference):** The final denoised latent representation is decoded back into pixel space using the VAE decoder to generate the final image.

```
[Text Prompt] --> [Text Encoder] --> [Text Embeddings] --+
|
v
[Initial Latent Noise] <--- [Scheduler] <--- [U-Net] <-- (+) <-- [Noisy Latent (t)]
^                     | | |            ^          |
|---------------------| | |------------|          |
|                     V V V                       |
[Decoded Image] <-- [VAE Decoder] <-- [Denoised Latent (0)] <-- [Final U-Net Output]
```

*(Simplified diagram of the inference process)*

## Features

* Core implementation of the U-Net architecture with attention mechanisms.
* VAE integration for latent space operations.
* CLIP text encoder integration for conditioning.
* Implementation of a noise scheduler (Specify which one, e.g., DDPM/DDIM).
* Training script for training the U-Net model.
* Inference script for generating images from text prompts.
* Modular design for easier understanding and modification.

## Prerequisites

* Python (e.g., 3.8+)
* PyTorch (e.g., 1.12+ or 2.0+)
* `transformers` (for pre-trained text encoder)
* `torchvision`
* `numpy`
* `Pillow` (PIL)
* `accelerate` (Recommended for efficient training, mixed precision, and multi-GPU support)
* `tqdm` (for progress bars)
* (Add any other specific libraries you use, e.g., `einops`, `wandb` for logging)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Make sure to create a `requirements.txt` file listing all dependencies)*

## Dataset Preparation

* The model expects a dataset of image-caption pairs.
* Describe the expected format (e.g., images in one folder, captions in a corresponding `.txt` file, or a metadata file like `metadata.csv` or `metadata.jsonl`).
* Mention the script or class responsible for loading the data (e.g., `dataset.py`).
* Example structure:
    ```bash
    dataset/
    ├── images/
    │   ├── image001.png
    │   ├── image002.jpg
    │   └── ...
    └── captions/
        ├── image001.txt
        ├── image002.txt
        └── ...
    ```
    *Or using a metadata file:*
    ```
    dataset/
    ├── images/
    │   ├── image001.png
    │   ├── image002.jpg
    │   └── ...
    └── metadata.csv
    ```
    *(Where `metadata.csv` might contain columns like `file_name,caption`)*

## Pre-trained Weights

For practical training and inference, this implementation relies on pre-trained weights for certain components:

* **VAE:** Download or specify the path to a pre-trained VAE model compatible with Stable Diffusion (e.g., from Hugging Face Hub: `stabilityai/sd-vae-ft-mse`).
* **Text Encoder:** Uses a pre-trained text encoder, typically CLIP ViT-L/14 (`openai/clip-vit-large-patch14`). The `transformers` library usually handles downloading these automatically.

*(Specify how users should obtain or configure paths to these weights in your configuration files or scripts)*

## Configuration

Model parameters, training settings, and paths are configured via [Specify method: e.g., YAML files, command-line arguments, constants in a config.py file].

Key configuration options include:

* `dataset_path`: Path to the prepared dataset.
* `vae_weights_path`: Path to pre-trained VAE weights (if loaded locally).
* `output_dir`: Directory to save checkpoints and logs.
* `image_size`: Resolution to train on (e.g., 256, 512).
* `batch_size`: Training batch size.
* `learning_rate`: Optimizer learning rate.
* `num_train_epochs`: Number of training epochs.
* `gradient_accumulation_steps`: Steps for gradient accumulation.
* `mixed_precision`: Enable mixed precision training (`fp16`, `bf16`, or `no`).
* `diffusion_steps`: Number of steps in the diffusion process (e.g., 1000).
* `unet_dim`: Base dimension for the U-Net channels.
* (Add other relevant parameters)

*(Provide an example config file or command-line usage if applicable)*

## Usage

### Training

To train the U-Net model:

```bash
# Example using accelerate for multi-GPU/mixed-precision
accelerate launch train.py \
    --config path/to/your_config.yaml \
    # Or provide arguments directly:
    # --dataset_path path/to/dataset \
    # --output_dir ./sd-model-output \
    # --image_size 512 \
    # --batch_size 4 \
    # --learning_rate 1e-5 \
    # --num_train_epochs 50 \
    # --mixed_precision fp16

## Inference

To generate images from text prompts using a trained checkpoint:

python inference.py \
    --prompt "A photo of an astronaut riding a horse on the moon" \
    --checkpoint_path path/to/your/unet_checkpoint.pt \
    --output_path generated_image.png \
    --image_size 512 \
    --guidance_scale 7.5 \
    --num_inference_steps 50 \
    --seed 42 \
    # (Add other inference parameters like negative prompt)

## Project Structure 
```bash
your-repo-name/
├── models/
│   ├── unet.py             # U-Net model definition
│   ├── vae.py              # VAE wrapper/loading (or definition if built from scratch)
│   └── text_encoder.py     # Text encoder wrapper/loading
├── diffusion/
│   └── scheduler.py        # Noise scheduler implementation
├── data/
│   └── dataset.py          # Dataset loading logic
├── configs/
│   └── base_config.yaml    # Example configuration file
├── scripts/                # Utility scripts (optional)
├── train.py                # Main training script
├── inference.py            # Image generation script
├── requirements.txt        # Project dependencies
└── README.md               # This file

```

TODO / Future Work

    [ ] Implement different noise schedulers (e.g., PNDM, Euler).
    [ ] Add support for classifier-free guidance during training.
    [ ] Integrate optimizations (e.g., xFormers for memory-efficient attention).
    [ ] Add support for fine-tuning on specific concepts.
    [ ] Implement different U-Net variants or sizes.
    [ ] Add evaluation metrics.
    [ ] Improve logging and visualization (e.g., TensorBoard, WandB).
