import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer
from data import load_and_tokenize_data
from model import load_model

def compute_metrics(eval_p):
    """
    Calculates accuracy from model predictions.
    """
    # Unpack the evaluation predictions: logits are raw model outputs, labels are true labels
    logits, labels = eval_p
    # Convert logits to predictions by selecting the class with the highest logit value
    predictions = np.argmax(logits, axis=-1)
    # Load the accuracy metric
    metric = evaluate.load("accuracy")
    # Compute and return the accuracy
    return metric.compute(predictions=predictions, references=labels)

def train_model(output_dir="Tuned Model", num_train_epochs=2, learning_rate=1e-5, per_device_train_batch_size=2, per_device_eval_batch_size=2):
    """
    Load data, model, and train the model with specified training arguments.
    """
    # Load and tokenize the training and evaluation datasets
    train_dataset, eval_dataset, tokenizer = load_and_tokenize_data()
    # Load the pretrained model
    model = load_model()

    # Define training arguments for fine-tuning
    training_args = TrainingArguments(
        output_dir=output_dir,  # Directory to save model checkpoints and logs
        evaluation_strategy="steps",  # Evaluate the model every few steps
        eval_steps=50,  # Number of steps between evaluations
        learning_rate=learning_rate,  # Learning rate for the optimizer
        per_device_train_batch_size=per_device_train_batch_size,  # Batch size for training
        per_device_eval_batch_size=per_device_eval_batch_size,  # Batch size for evaluation
        num_train_epochs=num_train_epochs,  # Number of training epochs
        weight_decay=0.01,  # Regularization to prevent overfitting
        logging_dir=output_dir,  # Directory for logging training progress
        save_strategy="no",  # Do not save checkpoints to save space
        no_cuda=True,  # Force CPU usage (no GPU)
        dataloader_num_workers=2,  # Number of CPU cores for data loading
    )

    # Initialize the Trainer with model, training args, datasets, and metrics
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Start the fine-tuning process
    print("Fine Tuning DistilBERT...")
    trainer.train()
    print("Training completed!")

    # Save the final fine-tuned model and tokenizer to the output directory
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    train_model()
