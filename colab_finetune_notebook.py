# Colab Notebook: GPT-2 Fine-Tuning for Email Generation
# Paste each section below into its own Colab cell, run in order with Shift+Enter.
# Add this install cell first (not shown below, run it as Cell 1):
#   !pip install transformers datasets accelerate

# ============================================================
# Section: Load Dataset — Topic/Tone/Email pairs (upload finetune_pairs.jsonl first)
# ============================================================
import json
from datasets import Dataset

# Upload finetune_pairs.jsonl to Colab first (left sidebar -> folder icon -> upload icon)
with open("finetune_pairs.jsonl") as f:
    raw_pairs = [json.loads(line) for line in f]

# IMPORTANT: this template must exactly match the prompt your app builds at
# inference time, or the fine-tuned model won't recognize the pattern.
def format_example(example):
    return (
        "Write a professional email based on the topic and tone given.\n\n"
        f"Topic: {example['topic']}\n"
        f"Tone: {example['tone']}\n"
        f"Email: {example['email']}"
    )

texts = [format_example(ex) for ex in raw_pairs]
dataset = Dataset.from_dict({"text": texts})

# Sanity-check the format
print(dataset[0]["text"])


# ============================================================
# Section: Load Tokenizer & Model
# ============================================================
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)
model.config.pad_token_id = tokenizer.pad_token_id


# ============================================================
# Section: Tokenization
# ============================================================
def tokenize_function(examples):
    # Append eos_token so the model learns WHERE an email ends
    texts_with_eos = [t + tokenizer.eos_token for t in examples["text"]]
    return tokenizer(
        texts_with_eos,
        truncation=True,
        padding="max_length",
        max_length=160
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Train/val split so you can measure whether fine-tuning actually helped
split = tokenized_dataset.train_test_split(test_size=0.15, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]


# ============================================================
# Section: Training Setup
# NOTE: overwrite_output_dir removed — newer transformers versions
# don't accept this argument and it's not needed (dir is overwritten by default)
# ============================================================
import torch
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=25,          # increased since dataset grew to ~39 examples
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    logging_steps=5,
    learning_rate=5e-5,
    fp16=torch.cuda.is_available(),
    report_to="none"
)


# ============================================================
# Section: Trainer
# ============================================================
from transformers import Trainer, DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator
)


# ============================================================
# Section: Train Model + check perplexity before/after
# ============================================================
import math

# Perplexity BEFORE fine-tuning (sanity baseline)
baseline_eval = trainer.evaluate()
print(f"Baseline perplexity (before training): {math.exp(baseline_eval['eval_loss']):.2f}")

trainer.train()

# Perplexity AFTER fine-tuning
final_eval = trainer.evaluate()
print(f"Final perplexity (after training): {math.exp(final_eval['eval_loss']):.2f}")
# Lower perplexity = model is less "surprised" by real email text = better fit


# ============================================================
# Section: Save Model
# ============================================================
trainer.save_model("fine_tuned_gpt2")
tokenizer.save_pretrained("fine_tuned_gpt2")


# ============================================================
# Section: Zip and download the model
# ============================================================
import shutil
shutil.make_archive("fine_tuned_gpt2", "zip", "fine_tuned_gpt2")

from google.colab import files
files.download("fine_tuned_gpt2.zip")


# ============================================================
# Section: Test Your Model — matches the app's actual prompt format
# ============================================================
prompt = (
    "Write a professional email based on the topic and tone given.\n\n"
    "Topic: leave approval\n"
    "Tone: Formal\n"
    "Email:"
)

inputs = tokenizer(prompt, return_tensors="pt")
input_len = inputs["input_ids"].shape[1]

output = model.generate(
    **inputs,
    max_new_tokens=60,
    num_return_sequences=1,
    do_sample=True,
    top_k=50,
    top_p=0.9,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
)

# Decode only the NEW tokens so you see just the generated email, not the prompt echoed back
new_tokens = output[0][input_len:]
print(tokenizer.decode(new_tokens, skip_special_tokens=True))