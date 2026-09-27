#!/usr/bin/env python3
"""
[código próprio] Confirma os 3 vencedores de hpsearch/ (arquitetura, RoPE, BPE) em orçamento
COMPLETO (15000 iterações, split 90/10 padrão — mesma metodologia de `resultados 1/2/3`, não o
k-fold curto da busca) — 4 variações: uma por eixo isolado + uma combinando os três. Ver
o README do projeto, seção de busca de hiperparâmetros, para o racional e os resultados da busca que geraram essas
configs.

As 4 variações (baseline = `resultados 3`: n_layer=6, n_head=6, n_embd=384, block_size=256,
pos_emb_type=learned, tokenização char, vocab_size=177):

    arch  : arquitetura vencedora (n_layer=4, n_head=8, n_embd=384, block_size=128, dropout=0.3),
            resto igual ao baseline (learned, char)
    rope  : arquitetura do baseline, pos_emb_type=rope
    bpe   : arquitetura do baseline, tokenização BPE(1500) treinada no corpus
    all3  : arquitetura vencedora + rope + BPE(1500) — os três juntos

**BPE usa learning_rate/warmup menores** (mesma correção de `run_tokenization_comparison.py` —
ver run_tokenization_comparison.py): o LR do char-level faz o BPE divergir num vocabulário ~8x maior.

Cada variação roda UMA VEZ (sem k-fold, sem repetição) até `--iters` (padrão 15000), com o mesmo
`eval_interval`/`eval_iters`/`always_save_checkpoint` de `resultados 3`. Sequencial (não paralelo
— evita disputar o mesmo device MPS). Gera um `train_log_<tag>.txt` completo por variação (mesmo
formato usado por `resultados 1/2/3`, pronto pra virar `resultados 4/` com o script de sempre).

Uso:
    python3 run_confirmation_variants.py                # as 4 variações, 15000 iters cada
    python3 run_confirmation_variants.py --smoke-test    # valida o pipeline em ~1 min
"""
import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bpe_tokenizer import encode as bpe_encode, train_bpe
from kfold_split import build_global_vocab
from patch_model_pos_emb import MODEL_PATCHES, TRAIN_PATCHES, apply_patches

HERE = Path(__file__).parent
STEP_RE = re.compile(r"step (\d+): train loss ([\d.]+|nan), val loss ([\d.]+|nan)")

BASELINE_ARCH = dict(n_layer=6, n_head=6, n_embd=384, block_size=256, dropout=0.2)
WINNER_ARCH = dict(n_layer=4, n_head=8, n_embd=384, block_size=128, dropout=0.3)


def build_variants(bpe_vocab_size: int) -> dict:
    """tag -> (arch, pos_emb_type, dataset, learning_rate, warmup_iters). Nome do dataset BPE
    inclui o vocab_size pra nunca apontar pra uma pasta que não bate com o tokenizer treinado
    (ex: rodar --smoke-test com vocab menor não pode acidentalmente ler dados de um vocab maior
    já preparado, ou vice-versa)."""
    bpe_dataset = f"machado_bpe{bpe_vocab_size}"
    return {
        "arch": (WINNER_ARCH, "learned", "machado_char", 1e-3, 100),
        "rope": (BASELINE_ARCH, "rope", "machado_char", 1e-3, 100),
        "bpe": (BASELINE_ARCH, "learned", bpe_dataset, 3e-4, 300),
        "all3": (WINNER_ARCH, "rope", bpe_dataset, 3e-4, 300),
    }


def detect_device():
    import torch
    if torch.cuda.is_available():
        return "cuda", True
    if torch.backends.mps.is_available():
        return "mps", False
    return "cpu", False


def prepare_bpe_dataset(nanogpt_dir: Path, vocab_size: int, sample_chars: int, force: bool):
    """Split 90/10 PADRÃO (não k-fold) codificado com BPE — mesmo método de
    data/machado_char/prepare.py (célula 13 do notebook), só trocando char->BPE."""
    out_dir = nanogpt_dir / "data" / f"machado_bpe{vocab_size}"
    if not force and (out_dir / "train.bin").exists():
        print(f"[bpe-dataset] já existe em {out_dir}")
        return out_dir

    import pickle
    import numpy as np

    input_txt = nanogpt_dir / "data" / "machado_char" / "input.txt"
    text = input_txt.read_text(encoding="utf-8")

    cache_path = HERE / f"bpe_cache_v{vocab_size}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        tokenizer = cached["tokenizer"]
        print(f"[bpe-dataset] reaproveitando tokenizer cacheado ({len(tokenizer['vocab'])} tokens)")
    else:
        base_chars = sorted(set(text))
        print(f"[bpe-dataset] treinando BPE (vocab_size={vocab_size})...")
        tokenizer = train_bpe(text, vocab_size=vocab_size, base_chars=base_chars, sample_chars=sample_chars)
        with open(cache_path, "wb") as f:
            pickle.dump({"vocab_size_target": vocab_size, "sample_chars": sample_chars, "tokenizer": tokenizer}, f)

    n = len(text)
    train_text = text[: int(n * 0.9)]
    val_text = text[int(n * 0.9):]
    train_ids = np.array(bpe_encode(train_text, tokenizer), dtype=np.uint16)
    val_ids = np.array(bpe_encode(val_text, tokenizer), dtype=np.uint16)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_ids.tofile(out_dir / "train.bin")
    val_ids.tofile(out_dir / "val.bin")
    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump({"vocab_size": len(tokenizer["vocab"])}, f)
    print(f"[bpe-dataset] {len(train_ids):,} tokens treino / {len(val_ids):,} tokens val -> {out_dir}".replace(",", "."))
    return out_dir


def build_config_file(arch, pos_emb_type, dataset, learning_rate, warmup_iters, max_iters,
                       device, compile_flag, out_dir_name):
    eval_interval = 500 if max_iters >= 5000 else max(50, max_iters // 4)
    eval_iters = 100 if max_iters >= 5000 else min(50, max(10, max_iters // 20))
    return f"""
out_dir = '{out_dir_name}'
eval_interval = {eval_interval}
eval_iters = {eval_iters}
log_interval = 10
always_save_checkpoint = False
wandb_log = False

dataset = '{dataset}'
gradient_accumulation_steps = 1
batch_size = 128
block_size = {arch['block_size']}

n_layer = {arch['n_layer']}
n_head = {arch['n_head']}
n_embd = {arch['n_embd']}
dropout = {arch['dropout']}

learning_rate = {learning_rate}
max_iters = {max_iters}
lr_decay_iters = {max_iters}
min_lr = {learning_rate / 10}
beta2 = 0.99
warmup_iters = {warmup_iters}

device = '{device}'
compile = {compile_flag}
pos_emb_type = '{pos_emb_type}'
"""


def run_variant(nanogpt_dir: Path, tag: str, variants: dict, max_iters: int, device: str, compile_flag: bool, log_dir: Path):
    arch, pos_emb_type, dataset, lr, warmup_iters = variants[tag]
    out_dir_name = f"hpsearch/out/confirm_{tag}"
    config_text = build_config_file(arch, pos_emb_type, dataset, lr, warmup_iters, max_iters,
                                     device, compile_flag, out_dir_name)
    config_path = nanogpt_dir / "config" / "confirm_tmp.py"
    config_path.write_text(config_text, encoding="utf-8")

    log_path = log_dir / f"train_log_{tag}.txt"
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(
            [sys.executable, "train.py", str(config_path.relative_to(nanogpt_dir))],
            cwd=str(nanogpt_dir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        logf.write(proc.stdout)
    elapsed = time.time() - t0

    matches = STEP_RE.findall(proc.stdout)
    if not matches:
        print(f"    [aviso] {tag} não produziu nenhuma linha 'step X: ...' — ver {log_path}")
        return None, None, elapsed
    _, train_loss_s, val_loss_s = matches[-1]
    diverged = any(v == "nan" for _, _, v in matches)
    finite_vals = [float(v) for _, _, v in matches if v != "nan"]
    best_val = min(finite_vals) if finite_vals else float("nan")
    train_loss = float(train_loss_s) if train_loss_s != "nan" else float("nan")
    val_loss = float(val_loss_s) if val_loss_s != "nan" else float("nan")
    if diverged:
        n_nan = sum(1 for _, _, v in matches if v == "nan")
        print(f"    [DIVERGIU] {tag}: {n_nan}/{len(matches)} avaliações deram nan "
              f"(melhor val loss antes de divergir: {best_val:.4f}) — ver {log_path}")
    return train_loss, val_loss, elapsed, best_val, len(matches), diverged


def load_results(results_csv: Path):
    done = {}
    if results_csv.exists():
        with open(results_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done[row["variante"]] = row
    return done


def append_result(results_csv: Path, row: dict, write_header: bool):
    with open(results_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nanogpt-dir", default=str(HERE.parent / "notebook" / "nanoGPT"))
    ap.add_argument("--iters", type=int, default=15000)
    ap.add_argument("--bpe-vocab-size", type=int, default=1500)
    ap.add_argument("--bpe-sample-chars", type=int, default=2_000_000)
    ap.add_argument("--force-bpe-dataset", action="store_true")
    ap.add_argument("--results-csv", default=str(HERE / "resultados_confirmacao.csv"))
    ap.add_argument("--only", nargs="+", choices=["arch", "rope", "bpe", "all3"], default=None,
                     help="rodar só estas variações (padrão: todas as 4)")
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    if args.smoke_test:
        args.iters = 20
        args.bpe_vocab_size = 300
        args.bpe_sample_chars = 200_000
        print("[smoke-test] rodando com valores mínimos só pra validar o pipeline "
              "(resultado NÃO tem valor científico)")

    nanogpt_dir = Path(args.nanogpt_dir).resolve()
    results_csv = Path(args.results_csv)
    log_dir = HERE / "confirm_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if not (nanogpt_dir / "data" / "machado_char" / "train.bin").exists():
        sys.exit("Não encontrei data/machado_char/train.bin — rode o notebook (treino real) "
                  "pelo menos até gerar o split 90/10 padrão antes desta confirmação.")

    print("[patch] garantindo que model.py/train.py suportam pos_emb_type...")
    apply_patches(nanogpt_dir / "model.py", MODEL_PATCHES, "model.py")
    apply_patches(nanogpt_dir / "train.py", TRAIN_PATCHES, "train.py")

    device, compile_flag = detect_device()
    print(f"[device] {device} (compile={compile_flag})")

    variants = build_variants(args.bpe_vocab_size)
    needs_bpe = any(variants[t][2].startswith("machado_bpe") for t in (args.only or variants.keys()))
    if needs_bpe:
        prepare_bpe_dataset(nanogpt_dir, args.bpe_vocab_size, args.bpe_sample_chars, args.force_bpe_dataset)

    done = load_results(results_csv)
    write_header = not results_csv.exists()

    tags = args.only or list(variants.keys())
    for tag in tags:
        arch, pos_emb_type, dataset, lr, warmup_iters = variants[tag]
        if tag in done:
            print(f"[{tag}] já em {results_csv}, pulado (apague a linha pra re-rodar)")
            continue
        print(f"\n=== variação '{tag}' — arch={arch}, pos_emb_type={pos_emb_type}, "
              f"dataset={dataset}, lr={lr}, warmup={warmup_iters}, {args.iters} iters ===")
        result = run_variant(nanogpt_dir, tag, variants, args.iters, device, compile_flag, log_dir)
        if result[0] is None:
            continue
        train_loss, val_loss, elapsed, best_val, n_evals, diverged = result
        row = {
            "variante": tag,
            "arch": str(arch),
            "pos_emb_type": pos_emb_type,
            "dataset": dataset,
            "learning_rate": lr,
            "max_iters": args.iters,
            "train_loss_final": train_loss,
            "val_loss_final": val_loss,
            "val_loss_melhor": best_val,
            "divergiu": diverged,
            "n_avaliacoes": n_evals,
            "elapsed_s": round(elapsed, 1),
        }
        append_result(results_csv, row, write_header)
        write_header = False
        print(f"  -> val loss final {val_loss} (melhor: {best_val}), {elapsed/60:.1f}min")

    print(f"\nResultados completos em: {results_csv}")
    print(f"Logs de treino completos (mesmo formato de resultados 1/2/3) em: {log_dir}/train_log_<variante>.txt")


if __name__ == "__main__":
    main()
