
# Colab Notebook: GPT-2 Fine-Tuning for Email Generation
# Replace sections 3-8 of your existing notebook with these cells.
# Sections 1 (install), 2 (imports), 9 (save), 10 (test) stay the same.

# ============================================================
# 🔹 3. Load Dataset — Topic/Tone/Email pairs (upload finetune_pairs.jsonl first)
# ============================================================
import json
from datasets import Dataset

# Upload finetune_pairs.jsonl to Colab first (left sidebar -> Files -> upload)
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

# Optional: print one to sanity-check the format
print(dataset[0]["text"])


# ============================================================
# 🔹 4. Load Tokenizer & Model
# ============================================================
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)
model.config.pad_token_id = tokenizer.pad_token_id


# ============================================================
# 🔹 5. Tokenization
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
# 🔹 6. Training Setup
# ============================================================
import torch
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=15,          # small dataset -> needs more epochs to learn the pattern
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
# 🔹 7. Trainer
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
# 🔹 8. Train Model + check perplexity before/after
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
# 🔹 9. Save Model (unchanged from your original notebook)
# ============================================================
trainer.save_model("fine_tuned_gpt2")
tokenizer.save_pretrained("fine_tuned_gpt2")


# ============================================================
# 🔹 10. Test Your Model — matches the app's actual prompt format
# ============================================================
prompt = (
    "Write a professional email based on the topic and tone given.\n\n"
    "Topic: request a laptop upgrade\n"
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
