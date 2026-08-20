import pickle
import sys

def clean_text(text: str, min_len: int = 2) -> str:
    pass

with open('models/best_model.pkl', 'rb') as f:
    bundle = pickle.load(f)
    print("Loaded successfully!", bundle.keys())
