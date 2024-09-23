import re


class LocalWordTokenizer:
    def __init__(self, vocab=None, raw_text=None):
        assert vocab is None or raw_text is None, \
            "Send either the vocab or raw_text as input. Raw text will be used to create the vocabulary."

        if raw_text:
            preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
            preprocessed = [item for item in preprocessed if item]
            all_words = sorted(set(preprocessed))
            vocab = {token: integer for integer, token in enumerate(all_words)}

        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
