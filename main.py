from fastapi import FastAPI, Request
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import torch
import numpy as np

app = FastAPI()

# Load model and tokenizer from Hugging Face
model = BertForSequenceClassification.from_pretrained("JaySenpai/bert-model")
tokenizer = BertTokenizer.from_pretrained("JaySenpai/bert-model")
model.eval()

# Load label encoder classes
le = LabelEncoder()
le.classes_ = np.load("label_classes.npy", allow_pickle=True)

@app.post("/predict")
async def predict(data: dict):
    text = data["text"]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_class = torch.argmax(probs, dim=1).item()
        pred_label = le.classes_[pred_class]
        confidence = probs[0][pred_class].item()
    return {"predicted_category": pred_label, "confidence": confidence}
