import re
from collections import defaultdict, Counter


class NGramModel:
    def __init__(self, n=2):
        self.n = n
        self.model = defaultdict(Counter)

    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text.split()

    def train(self, corpus):
        for line in corpus:
            tokens = self.preprocess(line)
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i:i + self.n - 1])
                next_word = tokens[i + self.n - 1]
                self.model[context][next_word] += 1

    def predict(self, text, top_k=3, max_chain=4):
        """
        Smart continuation: instead of appending just 1 word, chains up to
        `max_chain` words by repeatedly walking the n-gram model forward.
        """
        tokens = self.preprocess(text)
        if len(tokens) < self.n - 1:
            return []

        results = []
        seen_first_words = set()

        context = tuple(tokens[-(self.n - 1):])
        candidates = self.model.get(context, {})
        total = sum(candidates.values()) or 1

        for word, count in candidates.most_common(top_k):
            if word in seen_first_words:
                continue
            seen_first_words.add(word)

            chained_words = [word]
            walk_context = tuple(list(context[1:]) + [word]) if self.n > 1 else ()

            for _ in range(max_chain - 1):
                next_candidates = self.model.get(walk_context, {})
                if not next_candidates:
                    break
                next_word, _ = next_candidates.most_common(1)[0]
                chained_words.append(next_word)
                walk_context = tuple(list(walk_context[1:]) + [next_word]) if self.n > 1 else ()

            phrase = " ".join(chained_words)
            results.append((f"{text} {phrase}", count / total))

        return results
