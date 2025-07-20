from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import torch
import numpy as np

app = FastAPI()

model = BertForSequenceClassification.from_pretrained("JaySenpai/bert-model")
tokenizer = BertTokenizer.from_pretrained("JaySenpai/bert-model")
model.eval()

le = LabelEncoder()
le.classes_ = np.load(r"C:\Users\jayes\Downloads\label_classes.npy", allow_pickle=True)

class TextInput(BaseModel):
    text: str

@app.post("/predict")
async def predict(data: TextInput):
    text = data.text  # ✅ Dot notation
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_class = torch.argmax(probs, dim=1).item()
        pred_label = le.classes_[pred_class]
        confidence = probs[0][pred_class].item()
    return {"predicted_category": pred_label, "confidence": confidence}
