from datasets import load_dataset
from transformers import DistilBertTokenizerFast

data = load_dataset("imdb")
tok = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize_fn(X):
    return tok(X["text"], truncation = True, padding="max_length")

shuffle_val_data = data['test'].shuffle(seed=42)
tokenized_val_data = shuffle_val_data.map(tokenize_fn, batched=True)
small_val_data = tokenized_val_data.select(range(1000))
small_val_data = small_val_data.with_format("torch", columns=["label", "input_ids", "attention_mask"])

print(small_val_data)
print(small_val_data[0])
