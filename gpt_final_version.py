import math
import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH_SIZE = 64
BLOCK_SIZE = 256
MAX_ITERS = 5000
EVAL_INTERVAL = 500
LEARNING_RATE = 5e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_ITERS = 200
N_EMBD = 384
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.1
MAX_LENGTH = 5000

# ----------------------


class GenerateText:
    """
    Class method for getting the data and reading it
    """

    def __init__(self, filepath):
        """
        Initialization of the GenerateText() class
        """
        self.filepath = filepath
        self.text = self._read_file()
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)

        # used to encode and decode
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}

        # generating usable data
        self.data = torch.tensor(self.encode(self.text), dtype=torch.long)
        self.train_data, self.val_data = self._split_data(train_ratio=0.9)

    def _read_file(self):
        """
        Method to read file
        """
        with open(self.filepath, "r", encoding="utf-8") as f:
            return f.read()

    def _split_data(self, train_ratio=0.9):
        """
        Splits data into training and validation datasets
        """
        n = int(train_ratio * len(self.data))
        return self.data[:n], self.data[n:]

    def encode(self, x):
        """
        Encoding chracters in indexes
        """
        return [self.char_to_idx[i] for i in x]

    def decode(self, x):
        """
        Decoding indexes into strings
        """
        return "".join([self.idx_to_char[i] for i in x])

    def get_batch(self, split, block_size = BLOCK_SIZE, batch_size = BATCH_SIZE, device = DEVICE):
        """
        Used to get a batch of data for batch gradient descent
        """
        batch_data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(batch_data) - block_size, (batch_size,))
        xb = torch.stack([batch_data[i : i + block_size] for i in ix])
        yb = torch.stack([batch_data[i + 1 : i + block_size + 1] for i in ix])
        xb, yb = xb.to(device), yb.to(device)
        return xb, yb

    @torch.no_grad()
    def get_loss(self, model, eval_iterations = EVAL_ITERS):
        """
        Used to get the loss of the model
        """
        out = {}
        model.eval()
        for split in ["train", "eval"]:
            losses = torch.zeros(eval_iterations)
            for k in range(eval_iterations):
                x, y = self.get_batch(split)
                loss = model(x, y)[1]
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

class Head(nn.Module):
    """
    A single head of self-attention
    """
    def __init__(self, head_size: int, n_embd: int = N_EMBD, block_size: int = BLOCK_SIZE,\
                  dropout: float = DROPOUT):
        """
        Initialization of the Head Class
        """
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for the Head Module
        """
        t, c = x.shape[1:]
        q, k, v = self.query(x), self.key(x), self.value(x)

        # compute attention scores
        w = q @ k.transpose(-2, 1) * c ** -0.5
        wei = w.masked_fill(self.tril[:t, :t] == 0, float('-inf'))
        wei = F.softmax(wei, dim = -1)

        # perform weighted aggregiation using our values
        output = wei @ v
        return output

class MultiHeadedAttention(nn.Module):
    """
    Multiple Heads of Self-Attention in Parallel
    """
    def __init__(self, num_heads, head_size, n_embd = N_EMBD, dropout = DROPOUT):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for the MultiHeadedAttention class
        """
        out = torch.cat([h(x) for h in self.heads])
        out = self.proj(out)
        return out

class FeedForwardNetwork(nn.Module):
    """
    Feed Forward Network with ReLU activation in between
    """
    def __init__(self, n_embd = N_EMBD):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
        )

    def forward(self, x):
        """
        Forward pass for the feed forward network
        """
        return self.net(x)

class Block(nn.Module):
    """
    Block that encapsulates all the previous neural network layers into one
    """
    def __init__(self, n_embd = N_EMBD, n_head = N_HEAD):
        super().__init__()
        head_size = n_embd // n_head

        # self-attention layer
        self.sa = MultiHeadedAttention(n_embd, head_size)

        # feed forward network initalization
        self.ffwd = FeedForwardNetwork(n_embd)

        # initialization of the two layer norms used in the structure
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        """
        Forward pass for the Block Class
        """
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class PositionalEncoder(nn.Module):
    """
    Adds positional encoding functionality for the language model
    """
    def __init__(self, d_model, dropout = DROPOUT, max_len = MAX_LENGTH):