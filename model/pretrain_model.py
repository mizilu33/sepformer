

import os
import torch

from transformers import AutoTokenizer, AutoModel

currunt_dir = os.path.dirname(__file__) 

sen_trans_pretrained_path = os.path.join(currunt_dir, "pretrained_w", "sentence-transformers")
model_name = 'sentence-transformers/all-MiniLM-L6-v2'



tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=os.path.join(sen_trans_pretrained_path, "tokenizer"))

model = AutoModel.from_pretrained(
    model_name,
    cache_dir=os.path.join(sen_trans_pretrained_path, "model"))

# torch.save(model, os.path.join(sen_trans_pretrained_path, 'pretr_model.pt'))
