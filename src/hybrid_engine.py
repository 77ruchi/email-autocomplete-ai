class HybridEngine:
    def __init__(self, ngram, gpt2):
        self.ngram = ngram
        self.gpt2 = gpt2

    def predict(self, text, mode="Hybrid", tone="Formal"):
        word_count = len(text.split())
        prompt = self._apply_tone(text, tone)

        if mode == "NLP":
            return self._smart_append(text, self.ngram.predict(text))
        elif mode == "GPT-2":
            return self._smart_append(text, self.gpt2.generate(prompt))
        elif mode == "Hybrid":
            if word_count < 3:
                return self._smart_append(text, self.ngram.predict(text))
            return self._smart_append(text, self.gpt2.generate(prompt))
        return []

    def generate_full_email(self, topic: str, tone="Formal"):
        prompt, forced_opening = self._build_email_prompt(topic, tone)
        continuation = self.gpt2.generate_email(prompt, max_new_tokens=60)
        return f"{forced_opening} {continuation}".strip()

    def _apply_tone(self, text, tone):
        if tone == "Casual":
            return f"Hey, {text}" if not text.lower().startswith("hey") else text
        return text

    def _build_email_prompt(self, topic, tone):
        greeting = "Dear Team," if tone == "Formal" else "Hey,"
        forced_opening = f"{greeting} I am writing regarding {topic}."

        prompt = (
            "Write a professional email based on the topic and tone given.\n\n"
            "Topic: schedule a meeting\n"
            "Tone: Formal\n"
            "Email: Dear Team, I would like to schedule a meeting at your "
            "earliest convenience. Please let me know a time that works for "
            "you. Best regards,\n\n"
            "Topic: say thanks\n"
            "Tone: Casual\n"
            "Email: Hey, just wanted to say thanks so much for your help "
            "earlier! Really appreciate it. Thanks,\n\n"
            f"Topic: {topic}\n"
            f"Tone: {tone}\n"
            f"Email: {forced_opening}"
        )
        return prompt, forced_opening

    def _smart_append(self, original_text, engine_results):
        trimmed = original_text.rstrip()
        final = []
        for item in engine_results:
            continuation, score = item
            continuation = continuation.strip()
            if continuation.lower().startswith(trimmed.lower()):
                merged = continuation
            else:
                merged = self._merge_no_duplicate(trimmed, continuation)
            final.append((merged, score))
        return final

    def _merge_no_duplicate(self, base, continuation):
        base_words = base.split()
        cont_words = continuation.split()
        if not base_words or not cont_words:
            return f"{base} {continuation}".strip()
        if base_words[-1].lower() == cont_words[0].lower():
            cont_words = cont_words[1:]
        return f"{base} {' '.join(cont_words)}".strip()