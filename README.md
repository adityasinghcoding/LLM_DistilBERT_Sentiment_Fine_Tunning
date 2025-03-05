# **Fine-Tuning DistilBERT for Sentiment Analysis on IMDB Dataset**

Fine-tuning a pre-trained DistilBERT model for binary sentiment classification (positive/negative) on the IMDB movie review dataset. It provides a compact and efficient implementation for text classification using the Hugging Face Transformers library.

## *Features*
- Fine-tuning of the DistilBERT model, a lightweight and faster version of BERT.
- Sentiment analysis on the IMDB movie review dataset.
- CPU-based training for environments without GPU support.
- Custom tokenization and data preparation with padding, truncation, and max length.
- Integration of evaluation metrics (accuracy) for performance tracking during training.

 ## *Workflow*
### 1. *Load and Prepare Data*
- The IMDB dataset is loaded using the datasets library.
- Text data is tokenized using the pre-trained DistilBERT tokenizer with padding and truncation to handle text sequences efficiently.
- Subsets of the dataset are used for training (500 examples) and evaluation (100 examples) to speed up processing on a CPU.

### 2. *Model Setup*
- The pre-trained DistilBERT model with a classification head is loaded. It is configured to predict binary labels for sentiment analysis (positive/negative).

### 3. *Training Configuration*
- Training is configured using the TrainingArguments class:
  - Learning rate: 1e-5
  - Batch size: 2 (for training and evaluation)
  - Epochs: 2
  - Evaluation every 50 steps
  - CPU-based training (no_cuda=True).

### 4. *Evaluation*
- Accuracy is used as the evaluation metric.
- Predictions are computed by applying argmax on model logits.

### 5. *Training*
  - The Trainer class handles the training loop and evaluation.
  - The final tuned model and tokenizer are saved for deployment.

## *Files and Directories*
 - Outputs/: Stores training logs and evaluation results.
 - Tuned Model/: Stores the fine-tuned DistilBERT model and tokenizer.
