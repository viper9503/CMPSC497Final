import nltk
# download the punkt tokenizer for ROUGE
nltk.download("punkt", quiet=True)

import math
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate

def load_preds(model, tokenizer, file: str, max_in=128, max_out=512, batch_size=8):
    """
    Generates predictions and collects reference targets from a JSONL test file.
    """
    ds = load_dataset("json", data_files={"test": file})["test"]
    gen_texts, refs = [], []

    for start in tqdm(range(0, len(ds), batch_size), desc="Generating test split"):
        chunk = ds[start : start + batch_size]

        # tokenize prompts
        inputs = tokenizer(
            chunk["prompt"],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_in,
        )
        # move tensors to device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # generate
        outs = model.generate(
            **inputs,
            max_length=max_out,
            num_beams=4,
        )
        gen_texts += tokenizer.batch_decode(outs, skip_special_tokens=True)
        refs      += chunk["target"]

    return gen_texts, refs

def main():
    model_dir = "checkpoints/final"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔍 Loading model from {model_dir} on {device}")

    # load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)

    # generate predictions
    preds, refs = load_preds(model, tokenizer, "data/test.jsonl", batch_size=8)

    # compute ROUGE
    rouge = evaluate.load("rouge")
    rouge_scores = rouge.compute(predictions=preds, references=refs)
    print("✅ ROUGE scores:")
    for k, v in rouge_scores.items():
        print(f"  {k}: {v:.4f}")

    # compute average perplexity
    ppls = []
    for ref in tqdm(refs, desc="Computing PPL"):
        enc = tokenizer(ref, return_tensors="pt", truncation=True).input_ids.to(device)
        logits = model(enc, labels=enc).logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = enc[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="mean"
        )
        ppls.append(math.exp(loss.item()))

    avg_ppl = sum(ppls) / len(ppls)
    print(f"✅ Average PPL over {len(ppls)} examples: {avg_ppl:.2f}")

    # reload test split for printing samples
    ds = load_dataset("json", data_files={"test": "data/test.jsonl"})["test"]

    print("\n--- Sample generations ---")
    for prompt, gen, ref in zip(ds["prompt"][:3], preds[:3], refs[:3]):
        print("PROMPT:   ", prompt)
        print("GENERATED:", gen)
        print("REFERENCE:", ref)
        print("-" * 60)

if __name__ == "__main__":
    main()
