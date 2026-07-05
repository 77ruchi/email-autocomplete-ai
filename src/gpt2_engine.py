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
        """Used for LINE AUTOCOMPLETE: short multi-word continuations."""
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

    def generate_email(self, prompt, topic="", max_new_tokens=70, num_candidates=3):
        """Used for FULL EMAIL GENERATION: generates multiple candidates and
        picks the most coherent, substantial one."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_candidates,
                do_sample=True,
                top_k=40,
                top_p=0.85,
                temperature=0.7,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )

        candidates = []
        for out in outputs:
            new_tokens = out[input_len:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            text = self._clean_email(text)
            if text:
                candidates.append(text)

        if not candidates:
            return ""

        return self._pick_best_candidate(candidates, topic)

    def _pick_best_candidate(self, candidates, topic):
        """Prefers longer, more complete candidates over short ones that
        just repeat topic words verbatim without adding real content.
        Keyword overlap only breaks ties among similar-length candidates."""
        MIN_WORDS = 8

        substantial = [c for c in candidates if len(c.split()) >= MIN_WORDS]
        pool = substantial if substantial else candidates

        topic_words = set(topic.lower().split())

        def score(c):
            word_count = len(c.split())
            overlap = len(topic_words & set(c.lower().split())) if topic_words else 0
            return (word_count, overlap)

        return max(pool, key=score)

    def _clean_continuation(self, text):
        """Trims whitespace and cuts at sentence-ending punctuation."""
        text = text.strip()
        for stop in [".", "!", "?", "\n"]:
            if stop in text:
                text = text.split(stop)[0] + stop
                break
        return text

    def _clean_email(self, text):
        """Cuts off generation if the model starts repeating the instruction
        template, or starts a second email after finishing the first one."""
        text = text.strip()
        stop_phrases = [
            "Topic:", "Tone:", "Write a professional email",
            "\n\n\n", "Email:", "\nEmail"
        ]
        for stop_phrase in stop_phrases:
            if stop_phrase in text:
                text = text.split(stop_phrase)[0]
        return text.strip()