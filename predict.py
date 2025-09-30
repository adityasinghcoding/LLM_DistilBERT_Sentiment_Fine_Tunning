from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def load_model_and_tokenizer(model_path="Tuned Model"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

def predict_sentiment(tokenizer, model, text):
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probabilities).item()

    class_names = {
        0: "Negative",
        1: "Positive"
    }

    return {
        "text": text,
        "sentiment": class_names[predicted_class],
        "confidence": probabilities[0][predicted_class].item()
    }

def main():
    tokenizer, model = load_model_and_tokenizer()
    text = input("Enter text: ")
    result = predict_sentiment(tokenizer, model, text)
    print(f"Text: {result['text']}")
    print(f"Predicted sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']:.2%}")

if __name__ == "__main__":
    main()
