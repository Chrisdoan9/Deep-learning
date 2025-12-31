# Tiny GPT-style (decoder-only) language model from scratch in PyTorch
# pip install torch

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ----------------------------
# 1) Tiny dataset (real words)
# ----------------------------
sentences = [
    "i love machine learning",
    "i love deep learning",
    "i love bioinformatics",
    "deep learning uses transformers",
    "transformers learn from data",
    "i enjoy learning python",
    "python is great for bioinformatics",
]

PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"

def tokenize(s: str):
    return s.lower().split()

# Build vocab
all_tokens = [PAD, UNK, BOS, EOS]
for s in sentences:
    all_tokens += tokenize(s)
vocab = {w: i for i, w in enumerate(sorted(set(all_tokens)))}
id2word = {i: w for w, i in vocab.items()}

pad_id = vocab[PAD]
unk_id = vocab[UNK]
bos_id = vocab[BOS]
eos_id = vocab[EOS]

def encode_sentence(s: str):
    toks = [BOS] + tokenize(s) + [EOS]
    return [vocab.get(t, unk_id) for t in toks]

encoded = [encode_sentence(s) for s in sentences]
block_size = max(len(x) for x in encoded)  # context length (T)

def pad_to(x, L):
    return x + [pad_id] * (L - len(x))

# Next-token prediction:
# input_ids = tokens[:-1], target_ids = tokens[1:]
X = torch.tensor([pad_to(x[:-1], block_size - 1) for x in encoded], dtype=torch.long)
Y = torch.tensor([pad_to(x[1:],  block_size - 1) for x in encoded], dtype=torch.long)

# ----------------------------
# 2) Decoder-only model (GPT-ish)
# ----------------------------
class GPTConfig:
    def __init__(self, vocab_size, block_size, d_model=128, n_heads=4, n_layers=2, dropout=0.1):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.cfg = cfg
        self.head_dim = cfg.d_model // cfg.n_heads

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        # causal mask: (1, 1, T, T)
        T = cfg.block_size
        mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T)
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)

        # reshape to (B, nh, T, hd)
        q = q.view(B, T, self.cfg.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.cfg.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.cfg.n_heads, self.head_dim).transpose(1, 2)

        # attention scores (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # apply causal mask
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v  # (B, nh, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        y = self.resid_drop(self.proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.fc2 = nn.Linear(4 * cfg.d_model, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return self.drop(x)

class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx):
        # idx: (B, T)
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)  # (1, T)
        x = self.tok_emb(idx) + self.pos_emb(pos)  # (B, T, C)
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits

# ----------------------------
# 3) Train
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = GPTConfig(vocab_size=len(vocab), block_size=block_size - 1, d_model=128, n_heads=4, n_layers=2, dropout=0.1)
model = TinyGPT(cfg).to(device)

X = X.to(device)
Y = Y.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
optimizer = optim.AdamW(model.parameters(), lr=3e-4)

for epoch in range(1, 301):
    model.train()
    optimizer.zero_grad()
    logits = model(X)  # (B, T, V)
    loss = criterion(logits.reshape(-1, logits.size(-1)), Y.reshape(-1))
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch:03d} | loss={loss.item():.4f}")

# ----------------------------
# 4) Generate (real words)
# ----------------------------
@torch.no_grad()
def generate(prompt: str, max_new_tokens=12):
    model.eval()
    ids = [bos_id] + [vocab.get(t, unk_id) for t in tokenize(prompt)]
    # limit to block size
    ids = ids[-cfg.block_size:]

    for _ in range(max_new_tokens):
        idx = torch.tensor([ids], dtype=torch.long, device=device)  # (1, T)
        logits = model(idx)  # (1, T, V)
        next_logits = logits[0, -1, :]  # last position
        next_id = int(torch.argmax(next_logits).item())  # greedy decode
        ids.append(next_id)
        ids = ids[-cfg.block_size:]
        if next_id == eos_id:
            break

    words = [id2word[i] for i in ids]
    # remove BOS
    if words and words[0] == BOS:
        words = words[1:]
    return " ".join(words)

print("\n--- Samples ---")
print("Prompt: i enjoy")
print("Output:", generate("i enjoy"))

print("\nPrompt: transformers")
print("Output:", generate("transformers"))

print("\nPrompt: python is")
print("Output:", generate("python is"))