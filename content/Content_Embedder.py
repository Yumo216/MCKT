import json
import torch
from transformers import RobertaTokenizer, RobertaModel


tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base")
model.eval()
if torch.cuda.is_available():
    model.cuda()

with open("questions.json", "r", encoding="utf-8") as f:
    data = json.load(f)

question_texts = [v["content"] for v in data.values() if v.get("content")]


for i, text in enumerate(question_texts[:2]):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :]
    print(f"\nQuestion {i+1} Embedding (first 20 dims):\n", cls_emb.squeeze(0)[:20].cpu().numpy())
