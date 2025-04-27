import os, re, json
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from collections import defaultdict

# 1) Pull down the SciTLDR AIC split
print("Loading SciTLDR (AIC config)…")
raw = load_dataset(
    "allenai/scitldr",       # correct dataset path
    "AIC",                   # Abstract + Introduction + Conclusion
    split="train+validation+test"
)  # :contentReference[oaicite:0]{index=0}

# 2) Build prompt/target pairs by concatenating sentences
def clean(sentences):
    text = " ".join(sentences)
    text = re.sub(r"\s+", " ", text).strip()
    return text

examples = []
for ex in raw:
    prompt = clean(ex["source"])  # AIC sections
    target = clean(ex["target"])  # the TLDR summary
    # length filters (optional)
    if 20 < len(prompt.split()) < 300 and 10 < len(target.split()) < 100:
        examples.append({"prompt": prompt, "target": target})

print(f"Built {len(examples)} examples")

# 3) Split into train/val/test
train, temp = train_test_split(examples, test_size=0.2, random_state=42)
val, test  = train_test_split(temp,     test_size=0.5, random_state=42)

os.makedirs("data", exist_ok=True)
for split_name, split_data in [
    ("train", train),
    ("validation", val),
    ("test", test),
]:
    path = f"data/{split_name}.jsonl"
    print(f"Writing {len(split_data)} examples to {path}")
    with open(path, "w", encoding="utf-8") as f:
        for ex in split_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print("✅ data_prep.py finished — data/ directory populated")
