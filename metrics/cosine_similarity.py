import torch
import numpy as np

def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    last_hidden_state = outputs.hidden_states[-1]

    embedding = last_hidden_state[:, 0, :]

    return embedding

def cosine_similarity(gold, pred, tokenizer, model):
    gold_embedding = get_embedding(gold, tokenizer, model)
    pred_embedding = get_embedding(pred, tokenizer, model)

    similarity = np.dot(gold_embedding, pred_embedding.T) / (np.linalg.norm(gold_embedding) * np.linalg.norm(pred_embedding))

    return similarity.item()
