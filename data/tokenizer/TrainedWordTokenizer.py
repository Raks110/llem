import tiktoken


class TrainedWorkTokenizer:
    def __int__(self, encoding = "gpt2", allowed_special=None):

        if allowed_special is None:
            allowed_special = {"<|endoftext|>"}

        self.tokenizer = tiktoken.get_encoding(encoding)
        self.allowed_special = allowed_special

    def encode(self, text):
        return self.tokenizer.encode(text, allowed_special=self.allowed_special)

    def decode(self, ids):
        return self.tokenizer.decode(ids)
