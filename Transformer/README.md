# Character-Level GPT Transformer with Optuna Tuning

This project implements a character-level GPT-like Transformer model using PyTorch. It includes features for training the model on a text dataset, optimizing its hyperparameters using Optuna, and interacting with the trained model via a simple command-line chatbot interface.

## Features

*   **GPT Model:** Implementation of a decoder-only Transformer model based on GPT principles, including multi-head self-attention, feed-forward networks, and layer normalization.
*   **Character-Level Tokenization:** Simple tokenization based on the unique characters found in the training data.
*   **Hyperparameter Optimization:** Integrated hyperparameter tuning using the [Optuna](https://optuna.org/) library to find optimal settings for learning rate, embedding size, number of layers/heads, dropout, etc. Results are stored in an SQLite database (`gpt_tuning.db`).
*   **Efficient Data Loading:** Uses memory mapping (`mmap`) for potentially large datasets, loading random chunks during training.
*   **Training Script:** A comprehensive script (`gpt_train_optuna.py`) handles both Optuna tuning and final model training using the best-found parameters.
*   **Model Checkpointing:** Saves the best hyperparameters and the final trained model's state dictionary and configuration for later use.
*   **Chatbot Interface:** A separate script (`chatbot.py`) loads the trained model and allows interactive text generation in a command-line chat format.
*   **Device Agnostic:** Automatically uses CUDA GPU if available, otherwise falls back to CPU.

## Requirements

*   Python 3.8+
*   PyTorch (>= 1.8, check compatibility with your hardware/CUDA)
*   Optuna (`pip install optuna`)
*   NumPy (usually installed with PyTorch)

You can install the required libraries using pip:

```bash
pip install torch optuna
```

## Dataset Preparation

This project expects preprocessed training and validation data files, along with a vocabulary file.

 1. Obtain Raw Data: Download your desired text corpus (e.g., OpenWebText, Project Gutenberg texts).

 2.   Preprocess: You need to process your raw text data into the following files within a dataset/ subdirectory:
    * dataset/output_train.txt: A single large text file containing the training data.
    * dataset/output_val.txt: A single large text file containing the validation data (e.g., a 10% split from the original data).

    * dataset/vocab.txt: A text file where each line contains one unique character present in the training/validation data.

    * (Note: You might need a separate script to perform this consolidation and vocabulary extraction from your raw source data, like the .xz extraction script discussed previously.)

## Usage 

There are two main scripts:

1. gpt_train_optuna.py: For hyperparameter tuning and/or final model training.
2. chatbot.py: For interacting with a trained model.

## Running the chatbot

Make sure you have successfully trained a model and have the following files present:

    final_model_config.pkl

    final_model_state.pth

    dataset/vocab.txt

Run the chatbot script:
```bash 
python chatbot.py
```

    The script will load the model configuration and state dictionary.

    It will prompt you for input. Type your text and press Enter.

    The model will generate a continuation based on your prompt.

    Type quit or exit to end the session.

## Configuration

Key settings can be adjusted directly within the gpt_train_optuna.py script:

    File Paths: TRAIN_FILE, VAL_FILE, VOCAB_FILE, STUDY_DB_FILE, etc.

    Optuna Settings: ENABLE_OPTUNA, N_TRIALS, TUNING_MAX_ITERS, TUNING_EVAL_INTERVAL, TUNING_EVAL_ITERS.

    Final Training Settings: DEFAULT_MAX_ITERS.

    Default Hyperparameters: Default values like DEFAULT_BLOCK_SIZE, DEFAULT_BATCH_SIZE (if not tuned), DEFAULT_LR (if not tuned).

## File Structure

```
.
├── dataset/
│   ├── output_train.txt    # Preprocessed training data
│   ├── output_val.txt      # Preprocessed validation data
│   └── vocab.txt           # Vocabulary file
├── gpt_train_optuna.py     # Main script for tuning and training
├── chatbot.py              # Script for interacting with the trained model
├── gpt_tuning.db           # Optuna study database (generated)
├── best_params.pkl         # Best hyperparameters found by Optuna (generated)
├── final_model_config.pkl  # Configuration of the final trained model (generated)
├── final_model_state.pth   # Weights of the final trained model (generated)
└── README.md               # This file
```

## Future Work / Improvements

    Implement more sophisticated tokenization (e.g., SentencePiece, BPE).

    Experiment with larger models and datasets.

    Add more sophisticated sampling strategies (temperature, top-k, top-p) to the generate method.

    Improve the chatbot interface (e.g., using a web framework).

    Implement proper model checkpointing during the final long training run.

    Add learning rate scheduling.
