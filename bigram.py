import torch
import torch.nn as nn
from torch.nn import functional as F

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda string: [stoi[char] for char in string]
decode = lambda indices: "".join([itos[i] for i in indices])


data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


torch.manual_seed(1337)
batch_size = 4
block_size = 8

# x = train_data[:block_size]
# y = train_data[1 : block_size + 1]


def get_batch(split="train"):
    data = train_data if split == "train" else val_data
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


x_batch, y_batch = get_batch("train")


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets):
        logits = self.token_embedding_table(idx)
        # loss = F.cross_entropy(logits, targets)
        return logits


model = BigramLanguageModel(vocab_size)
output = model(x_batch, y_batch)
print(output.shape)  # Should be (batch_size, block_size, vocab_size)
