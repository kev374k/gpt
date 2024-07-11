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

    def get_batch(self, split):
        """
        Used to get a batch of data for batch gradient descent
        """
        batch_data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(batch_data) - BLOCK_SIZE, (BATCH_SIZE,))
        xb = torch.stack([batch_data[i : i + BLOCK_SIZE] for i in ix])
        yb = torch.stack([batch_data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        return xb, yb

    @torch.no_grad()
    def get_loss(self, model):
        """
        Used to get the loss of the model
        """
        out = {}
        model.eval()
        for split in ["train", "eval"]:
            losses = torch.zeros(EVAL_ITERS)
            for k in range(EVAL_ITERS):
                x, y = self.get_batch(split)
                loss = model(x, y)[1]
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out
