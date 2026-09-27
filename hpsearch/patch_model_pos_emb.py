#!/usr/bin/env python3
"""
[código próprio] Patch idempotente em `nanoGPT/model.py` e `train.py` (código de terceiros,
Karpathy) pra suportar 4 estratégias de embedding posicional, comparadas em
`run_embedding_comparison.py`:

  - 'learned'    : embedding de posição aprendida (padrão original do GPT-2/nanoGPT) — baseline.
  - 'sinusoidal' : tabela fixa senoidal (Vaswani et al., 2017, "Attention Is All You Need") —
                   sem parâmetros treináveis.
  - 'rope'       : Rotary Position Embedding (Su et al., 2021, "RoFormer") — posição codificada
                   dentro da atenção (rotaciona q/k), não é somada ao embedding de token.
  - 'none'       : sem nenhuma informação de posição — baseline de piso, pra quantificar quanto
                   a posição realmente contribui nesse corpus/tarefa.

Mesmo espírito dos patches já aplicados no notebook (célula de clone, seção 0: device_type/
GradScaler) — mudança mínima, documentada, e que preserva o comportamento original quando
`pos_emb_type='learned'` (default), então não afeta em nada o pipeline principal do notebook.

Uso:
    python3 patch_model_pos_emb.py [--nanogpt-dir ../notebook/nanoGPT]

Seguro rodar mais de uma vez (idempotente — cada trecho só é aplicado se ainda não foi).
"""
import argparse
import ast
from pathlib import Path

HERE = Path(__file__).parent

MODEL_PATCHES = [
    (
        "import torch\nimport torch.nn as nn\nfrom torch.nn import functional as F\n\nclass LayerNorm(nn.Module):",
        '''import torch
import torch.nn as nn
from torch.nn import functional as F


def _sinusoidal_table(max_len, dim):
    """[código próprio] Tabela fixa de posição senoidal (Vaswani et al., 2017, "Attention Is
    All You Need") — sem parâmetros treináveis, ao contrário do wpe aprendido padrão do GPT-2."""
    pe = torch.zeros(max_len, dim)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def _apply_rope(x, cos, sin):
    """[código próprio] Aplica Rotary Position Embedding (RoPE — Su et al., 2021, "RoFormer")
    a q ou k. x: (B, nh, T, hd); cos/sin pré-computados até block_size, forma (block_size, hd/2)."""
    T = x.size(-2)
    cos = cos[:T].unsqueeze(0).unsqueeze(0)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.stack([rx1, rx2], dim=-1).flatten(-2)

class LayerNorm(nn.Module):''',
    ),
    (
        "        self.dropout = config.dropout\n        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0",
        '''        self.dropout = config.dropout
        # [código próprio] RoPE: pré-computa cos/sin até block_size (buffer, não treinável)
        self.pos_emb_type = getattr(config, 'pos_emb_type', 'learned')
        if self.pos_emb_type == 'rope':
            head_dim = config.n_embd // config.n_head
            assert head_dim % 2 == 0, "RoPE exige head_dim par (n_embd/n_head)"
            inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
            t = torch.arange(config.block_size).float()
            freqs = torch.outer(t, inv_freq)
            self.register_buffer("rope_cos", torch.cos(freqs), persistent=False)
            self.register_buffer("rope_sin", torch.sin(freqs), persistent=False)
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0''',
    ),
    (
        "        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)\n\n        # causal self-attention;",
        '''        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # [código próprio] RoPE rotaciona q/k conforme a posição, antes do produto interno
        if self.pos_emb_type == 'rope':
            q = _apply_rope(q, self.rope_cos, self.rope_sin)
            k = _apply_rope(k, self.rope_cos, self.rope_sin)

        # causal self-attention;''',
    ),
    (
        """@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster""",
        """@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    # [código próprio] 'learned' (padrão GPT-2) | 'sinusoidal' | 'rope' | 'none' — ver
    # Projeto-1/hpsearch/run_embedding_comparison.py
    pos_emb_type: str = 'learned'""",
    ),
    (
        """        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)""",
        """        # [código próprio] wpe (embedding aprendida) só existe pra pos_emb_type='learned';
        # 'sinusoidal' usa um buffer fixo, 'rope'/'none' não somam nada ao embedding de token.
        pos_emb_type = getattr(config, 'pos_emb_type', 'learned')
        _transformer_dict = dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        )
        if pos_emb_type == 'learned':
            _transformer_dict['wpe'] = nn.Embedding(config.block_size, config.n_embd)
        self.transformer = nn.ModuleDict(_transformer_dict)
        if pos_emb_type == 'sinusoidal':
            self.register_buffer('pos_emb_fixed', _sinusoidal_table(config.block_size, config.n_embd), persistent=False)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)""",
    ),
    (
        """        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params""",
        """        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and 'wpe' in self.transformer:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params""",
    ),
    (
        """        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)""",
        """        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        # [código próprio] soma de posição depende de pos_emb_type — 'rope' é tratado dentro
        # da atenção (CausalSelfAttention), 'none' não adiciona nenhuma informação de posição
        pos_emb_type = getattr(self.config, 'pos_emb_type', 'learned')
        if pos_emb_type == 'learned':
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        elif pos_emb_type == 'sinusoidal':
            x = self.transformer.drop(tok_emb + self.pos_emb_fixed[:t])
        else:  # 'rope' ou 'none'
            x = self.transformer.drop(tok_emb)""",
    ),
    (
        "        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])",
        "        if 'wpe' in self.transformer:\n            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])",
    ),
]

TRAIN_PATCHES = [
    (
        "n_embd = 768\ndropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+\nbias = False # do we use bias inside LayerNorm and Linear layers?",
        "n_embd = 768\ndropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+\nbias = False # do we use bias inside LayerNorm and Linear layers?\npos_emb_type = 'learned' # [código próprio] 'learned' | 'sinusoidal' | 'rope' | 'none' — ver hpsearch/",
    ),
    (
        "model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,\n                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line",
        "model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,\n                  bias=bias, vocab_size=None, dropout=dropout, pos_emb_type=pos_emb_type) # start with model_args from command line",
    ),
    (
        "    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:\n        model_args[k] = checkpoint_model_args[k]",
        "    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'pos_emb_type']:\n        model_args[k] = checkpoint_model_args[k]",
    ),
]


def _neuter_for_ast(src: str) -> str:
    def neuter(l):
        s = l.strip()
        if s.startswith("!") or s.startswith("%"):
            indent = l[: len(l) - len(l.lstrip())]
            return indent + "pass"
        return l
    return "\n".join(neuter(l) for l in src.split("\n"))


def apply_patches(path: Path, patches: list[tuple[str, str]], label: str) -> None:
    src = path.read_text(encoding="utf-8")
    applied, already = 0, 0
    for old, new in patches:
        # checa "já aplicado" ANTES de "original" — em alguns patches `new` contém `old` como
        # substring (ex: um campo novo acrescentado ao final de uma classe já existente), então
        # checar na ordem errada reaplicaria o patch a cada execução (não seria idempotente)
        if new in src:
            already += 1
        elif old in src:
            src = src.replace(old, new, 1)
            applied += 1
        else:
            raise ValueError(
                f"[{label}] nenhum dos trechos (nem antigo nem já patcheado) encontrado em {path} "
                "— o arquivo pode ter mudado de versão; revise os patches antes de continuar."
            )
    ast.parse(_neuter_for_ast(src))  # garante que o resultado é Python válido antes de gravar
    path.write_text(src, encoding="utf-8")
    print(f"[{label}] {applied} trecho(s) aplicado(s), {already} já estava(m) patcheado(s) -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nanogpt-dir", default=str(HERE.parent / "notebook" / "nanoGPT"))
    args = ap.parse_args()

    nanogpt_dir = Path(args.nanogpt_dir).resolve()
    apply_patches(nanogpt_dir / "model.py", MODEL_PATCHES, "model.py")
    apply_patches(nanogpt_dir / "train.py", TRAIN_PATCHES, "train.py")


if __name__ == "__main__":
    main()
