from transformers import AutoModelForSequenceClassification

def load_model(model_name="distilbert-base-uncased", num_labels=2):
    """
    Load a pretrained model for sequence classification.

    Args:
        model_name (str): Pretrained model name.
        num_labels (int): Number of output labels/classes.

    Returns:
        model: The loaded model instance.
    """
    # Load the DistilBERT model with a classification head for the specified number of labels
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    return model
