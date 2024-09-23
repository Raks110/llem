import torch
from torch.utils.data import Dataset, DataLoader


class DataSampler(Dataset):
    def __init__(self,
                 raw_text,
                 tokenizer,
                 batch_size=8,
                 max_length=4,
                 stride=4,
                 shuffle=False,
                 drop_last=True,
                 num_workers=0):
        self.target_ids = []
        self.input_ids = []
        token_ids = tokenizer.encode(raw_text, allowed_special={"<|endoftext|>"})
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + stride: i + max_length + max_length]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

        dataloader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers
        )
        self.data_iter = iter(dataloader)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

    def next(self):
        return next(self.data_iter)
