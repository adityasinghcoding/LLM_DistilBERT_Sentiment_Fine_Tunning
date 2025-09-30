from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_tokenize_data(tokenizer_name="distilbert-base-uncased", max_length=128, train_size=500, eval_size=100, seed=42):
    """
    Load the IMDB dataset and tokenize it using the specified tokenizer.

    Args:
        tokenizer_name (str): Pretrained tokenizer name.
        max_length (int): Maximum sequence length for padding/truncation.
        train_size (int): Number of training samples to select.
        eval_size (int): Number of evaluation samples to select.
        seed (int): Random seed for shuffling.

    Returns:
        train_dataset, eval_dataset: Tokenized and subsetted datasets.
        tokenizer: The tokenizer instance.
    """
    # Load the IMDB movie review dataset with positive and negative sentiment labels
    dataset = load_dataset("imdb")

    # Initialize the tokenizer for DistilBERT model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Define a function to tokenize the text examples
    def tokenize_function(examples):
        # Tokenize the text with padding and truncation to max_length
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    # Apply the tokenizer to the entire dataset in batches
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Shuffle and select a smaller subset of the training data for faster training
    train_dataset = tokenized_dataset['train'].shuffle(seed=seed).select(range(train_size))
    # Shuffle and select a smaller subset of the evaluation data
    eval_dataset = tokenized_dataset['test'].shuffle(seed=seed).select(range(eval_size))

    # Return the processed datasets and tokenizer
    return train_dataset, eval_dataset, tokenizer
