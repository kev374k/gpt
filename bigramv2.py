import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = 32
block_size = 8
max_iters = 10000
eval_interval = 500
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 32

with open("gpt/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for ch, i in char_to_idx.items()}


def encode(x):
    """
    encoding the characters into indexes
    """
    return [char_to_idx[i] for i in x]


def decode(x):
    """
    decoding integers into chars into string
    """
    return "".join([idx_to_char[i] for i in x])


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    xb = torch.stack([data[i : i + block_size] for i in ix])
    yb = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    xb, yb = xb.to(device), yb.to(device)
    return xb, yb


@torch.no_grad()
def get_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """
    head of self-attention
    """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.query(x), self.key(x), self.value(x)
    
        # compute attention scores
        w = q @ k.transpose(-2, -1) * C ** -0.5
        wei = w.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim = -1)
    
        # perform weighteted aggregation using v from before
        output = wei @ v
        return output

class MultiHeadAttention(nn.Module):
    """
    multiple heads of self-attention in parallel
    """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim = -1)

class BigramLanguageModel(nn.Module):
    """
    initialization of the BigramLanguageModel
    """
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.sa_head = MultiHeadAttention(8, n_embd//8)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensors
        token_embeddings = self.token_embedding_table(idx) # (B, T, C)
        positional_embeddings = self.position_embedding_table(torch.arange(T, device = device))
        x = token_embeddings + positional_embeddings
        x = self.sa_head(x)
        logits = self.lm_head(x) #(B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            A, B, C = logits.shape
            logits = logits.view(A * B, C)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond =  idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iteration in range(max_iters):
    if iteration % eval_interval == 0:
        losses = get_loss()
        print(
            f"step {iteration}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
