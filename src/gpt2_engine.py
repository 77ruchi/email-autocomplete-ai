import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class GPT2Engine:
    def __init__(self, model_path="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()

    def generate(self, text, max_new_tokens=6, num_return_sequences=3):
        inputs = self.tokenizer(text, return_tensors="pt")
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )

        results = []
        for out in outputs:
            new_tokens = out[input_len:]
            continuation = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            continuation = self._clean_continuation(continuation)
            if continuation:
                results.append((continuation, 1.0))
        return results

    def generate_email(self, prompt, max_new_tokens=70):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                do_sample=True,
                top_k=40,
                top_p=0.85,
                temperature=0.7,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )

        new_tokens = output[0][input_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return self._clean_email(text)

    def _clean_continuation(self, text):
        text = text.strip()
        for stop in [".", "!", "?", "\n"]:
            if stop in text:
                text = text.split(stop)[0] + stop
                break
        return text

    def _clean_email(self, text):
        text = text.strip()
        for stop_phrase in ["Topic:", "Tone:", "Write a professional email", "\n\n\n"]:
            if stop_phrase in text:
                text = text.split(stop_phrase)[0]
        return text.strip()