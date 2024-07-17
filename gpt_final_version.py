import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH_SIZE = 24
BLOCK_SIZE = 128
MAX_ITERS = 5000
EVAL_INTERVAL = 500
LEARNING_RATE = 5e-4
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
EVAL_ITERS = 10
N_EMBD = 128
N_HEAD = 16
N_LAYER = 16
DROPOUT = 0.2
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
        self.data = torch.tensor(self.encode(self.text), dtype=torch.int32)
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

    def get_batch(
        self, split, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE, device=DEVICE
    ):
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
    def get_loss(self, model, eval_iterations=EVAL_ITERS):
        """
        Used to get the loss of the model
        """
        out = {}
        model.eval()
        print("working")
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iterations)
            for k in range(eval_iterations):
                x, y = self.get_batch(split)
                logits, loss = model(x, y)
                print(loss)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out


class LayerNorm(nn.Module):
    """
    LayerNorm w/optional bias bc PyTorch doesn't support when bias = None
    """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        """
        Forward Pass
        """
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class SelfAttention(nn.Module):
    """
    Full Layers of Self Attention
    """

    def __init__(self, config):
        """
        Initialization of the Head Class
        """
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.block_size = config.block_size
        self.dropout = config.dropout
        # getting the key, query, and value projections for all heads in a batch
        self.c_attention = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)

        # output projection
        self.c_projection = nn.Linear(self.n_embd, self.n_embd)

        # regularization
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            print("Using Slow Attention, Flash requires PyTorch >= 2.0")
            self.register_buffer(
                "bias", torch.tril(torch.ones(self.block_size, self.block_size))
            ).view(1, 1, self.block_size, self.block_size)

    def forward(self, x):
        """
        Forward pass for the Head Module
        """
        b, t, c = x.shape
        q, k, v = self.c_attention(x).split(self.n_embd, dim=-1)
        q = q.view(b, t, self.n_head, c // self.n_head).transpose(
            1, 2
        )  # (B, nH, T, hs)
        k = k.view(b, t, self.n_head, c // self.n_head).transpose(
            1, 2
        )  # (B, nH, T, hs)
        v = v.view(b, t, self.n_head, c // self.n_head).transpose(
            1, 2
        )  # (B, nH, T, hs)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True
            )
        else:
            # compute attention scores
            w = q @ k.transpose(-2, 1) * c**-0.5
            wei = w.masked_fill(self.bias[:t, :t] == 0, float("-inf"))
            wei = F.softmax(wei, dim=-1)

            # perform weighted aggregiation using our values
            y = wei @ v

        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.resid_dropout(self.c_projection(y))
        return y


class FeedForwardNetwork(nn.Module):
    """
    Feed Forward Network with ReLU activation in between
    """

    def __init__(self, config):
        super().__init__()
        n_embd = config.n_embd
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.GELU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(config.dropout),
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

    def __init__(self, config):
        super().__init__()
        # self-attention layer
        self.sa = SelfAttention(config)

        # feed forward network initalization
        self.ffwd = FeedForwardNetwork(config)

        # initialization of the two layer norms used in the structure
        self.ln1 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln2 = LayerNorm(config.n_embd, bias=config.bias)

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

    def __init__(self, d_model, config):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)
        max_len = config.max_len
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Forward pass for the PositionalEncoder Cl ass
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


@dataclass
class GPTConfig:
    """
    Data Configurations for GPT
    """

    file_path: str
    block_size: int = BLOCK_SIZE
    n_layer: int = N_LAYER
    n_head: int = N_HEAD
    n_embd: int = N_EMBD
    max_len: int = MAX_LENGTH
    dropout: int = DROPOUT
    bias: bool = False
    device: str = DEVICE

    def __post_init__(self):
        self.text_generator = GenerateText(self.file_path)
        self.vocab_size = self.text_generator.vocab_size


class GPT(nn.Module):
    """
    Initialization of the GPT
    """

    def __init__(self, config):
        super().__init__()
        self.transformer = nn.ModuleDict(
            dict(
                # token embedding table
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                # positional embedding table
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"Number of Parameters: {self.get_num_params() / 1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters used in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return int(n_params)

    def forward(self, idx, targets=None):
        """
        Forward Pass of the GPT
        """
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.transformer.wte(idx)  # (B,T,C)
        pos_emb = self.transformer.wpe(torch.arange(T, device=DEVICE))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)\
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, config, max_new_tokens):
        """
        Used to generate new text, up to a maximum
        """
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -config.block_size :]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


class ModelHandler:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def save_model(self, model_save_path):
        """
        Used to save the model when it is done training
        """
        torch.save(self.model.state_dict(), model_save_path)

    def load_model(self, model_save_path):
        """
        Loading a pre-saved model
        """
        self.model.load_state_dict(torch.load(model_save_path, weights_only=True))
        self.model.eval()
        return self.model

    def train_model(
        self, eval_interval=EVAL_INTERVAL, lr=LEARNING_RATE, max_iters=MAX_ITERS
    ):
        """
        Method to train model
        """
        self.model.to(self.config.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr)

        for iteration in range(max_iters):
            if iteration % eval_interval == 0 or iteration == max_iters - 1:
                losses = textGenerator.get_loss(model=model)
                print(
                    f"Step {iteration}: Train Loss {losses['train']:.4f}, Val Loss {losses['val']:.4f}"
                )
            xb, yb = textGenerator.get_batch("train")

            cur_logits, cur_loss = model(xb, yb)

            optimizer.zero_grad(set_to_none=True)
            cur_loss.backward()
            optimizer.step()

    def generate_tokens(self, text_generator, max_token_length=1000):
        """
        Method used to generate new tokens
        """
        context = torch.zeros((1, 1), dtype=torch.int32, device=self.config.device)
        print(
            text_generator.decode(
                self.model.generate(context, self.config, max_token_length)[0].tolist()
            )
        )


FILE_PATH = "gpt/data/little_shakespeare.txt"
MODEL_PATH = "gpt/models/model1.pt"
configurations = GPTConfig(FILE_PATH)
textGenerator = GenerateText(FILE_PATH)
model = GPT(configurations)
modelHandler = ModelHandler(model, configurations)
modelHandler.train_model(max_iters=2000)
modelHandler.generate_tokens(textGenerator)
modelHandler.save_model(MODEL_PATH)


# context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
# print(
#     textGenerator.decode(
#         m.generate(context, config=configurations, max_new_tokens=1000)[0].tolist()
#     )
# )

# optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# for iteration in range(MAX_ITERS):
#     if iteration % EVAL_INTERVAL == 0 or iteration == MAX_ITERS - 1:
#         losses = textGenerator.get_loss(model=model)
#         print(
#             f"Step {iteration}: Train Loss {losses['train']:.4f}, Val Loss {losses['val']:.4f}"
#         )
#     xb, yb = textGenerator.get_batch("train")

#     cur_logits, cur_loss = model(xb, yb)

#     optimizer.zero_grad(set_to_none=True)
#     cur_loss.backward()
#     optimizer.step()
# context = torch.zeros((1, 1), dtype=torch.int32, device=DEVICE)
# print(
#     textGenerator.decode(
#         m.generate(context, config=configurations, max_new_tokens=1000)[0].tolist()
#     )
# )
