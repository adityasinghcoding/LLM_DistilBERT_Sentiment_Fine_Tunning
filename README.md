# Fine-Tuning DistilBERT for Sentiment Analysis on IMDB Dataset

This project fine-tunes a pre-trained DistilBERT model for binary sentiment classification (positive/negative) on the IMDB movie review dataset. It provides a modular and efficient implementation for text classification using the Hugging Face Transformers library.

## Features
- Fine-tuning of the DistilBERT model, a lightweight and faster version of BERT.
- Sentiment analysis on the IMDB movie review dataset.
- CPU-based training for environments without GPU support.
- Custom tokenization and data preparation with padding, truncation, and max length.
- Integration of evaluation metrics (accuracy) for performance tracking during training.
- Modular code structure with separate scripts for data loading, model setup, training, and prediction.
- Interactive prediction script for real-time sentiment analysis.

## Installation
Ensure you have Python 3.7+ installed. Install the required libraries using pip:

```
pip install transformers datasets evaluate numpy torch
```

## Usage

### Training the Model
Run the training script to fine-tune the model:

```
python train.py
```

This will load the IMDB dataset, tokenize it, train the DistilBERT model, and save the fine-tuned model and tokenizer to the `Tuned Model/` directory.

### Making Predictions
After training, use the prediction script to analyze sentiment on new text:

```
python predict.py
```

Enter the text when prompted, and the script will output the predicted sentiment (Positive/Negative) along with confidence score.

## Workflow
1. **Data Loading and Preparation** (`data.py`): Loads the IMDB dataset and tokenizes text using DistilBERT tokenizer with padding and truncation.
2. **Model Setup** (`model.py`): Loads the pre-trained DistilBERT model configured for binary classification.
3. **Training** (`train.py`): Configures training arguments (learning rate: 1e-5, batch size: 2, epochs: 2, CPU-only) and fine-tunes the model using Hugging Face Trainer.
4. **Evaluation**: Computes accuracy during training.
5. **Prediction** (`predict.py`): Loads the trained model and tokenizer to predict sentiment on user-input text.

## Project Structure
```
.
│   .gitattributes
│   data.py
│   LLM(DistilBert) Fine Tunning.ipynb
│   model.py
│   model_testing.ipynb
│   predict.py
│   README.md
│   train.py
│
├───Tokenizer
│       special_tokens_map.json
│       tokenizer.json
│       tokenizer_config.json
│       vocab.txt
│
├───Training Logs
│       events.out.tfevents.1741143242.Aditya00King.5992.0
│       events.out.tfevents.1741143686.Aditya00King.5992.1
│
└───Tuned Model
        config.json
        model.safetensors
```

## Files and Directories
- `data.py`: Handles dataset loading and tokenization.
- `model.py`: Loads the DistilBERT model for sequence classification.
- `train.py`: Script to train the model.
- `predict.py`: Script for sentiment prediction on new text.
- `LLM(DistilBert) Fine Tunning.ipynb`: Jupyter notebook version of the training process.
- `model_testing.ipynb`: Notebook for testing the model.
- `Tokenizer/`: Saved tokenizer files (tokenizer_config.json, vocab.txt, etc.).
- `Training Logs/`: TensorBoard logs from training (events.out.tfevents files).
- `Tuned Model/`: Fine-tuned model files (config.json, model.safetensors).
- `.gitattributes`: Git attributes for LFS tracking of large model files.
