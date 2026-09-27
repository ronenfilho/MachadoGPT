#!/usr/bin/env python3
"""
[código próprio] Comparação de estratégias de tokenização — char-level (atual, `resultados
1/2/3/`) vs. BPE pequeno treinado no próprio corpus (`bpe_tokenizer.py`) — arquitetura e demais
hiperparâmetros FIXOS na melhor config conhecida. Ver a seção "Busca de
hiperparâmetros", pro racional completo.

**Por que não dá pra comparar direto pelo val loss:** cross-entropy é medida em nats POR TOKEN,
e um token BPE cobre, em média, mais de um caractere (compressão) — um val loss de BPE mais
baixo que o do char-level não significa necessariamente um modelo melhor, pode só significar
"menos tokens pra errar". A métrica normalizada e comparável é **bits-por-caractere (BPC)**:

    BPC = (val_loss_nats / ln(2)) * (nº_tokens_val / nº_caracteres_val)

Pra char-level, tokens==caracteres, então BPC = val_loss_nats/ln(2) (a mesma perplexidade de
sempre, só em bits). Pra BPE, a razão tokens/caracteres é < 1 (compressão), o que puxa o BPC
pra baixo mesmo com val loss nominal mais alto — essa é a comparação justa.

BPE treinado com vocabulário pequeno (padrão 1500) de propósito: um vocabulário GPT-2 completo
(50257, via tiktoken) infralaria a tabela de embeddings pra ~19M parâmetros, maior que o resto
do modelo — deixaria de ser uma comparação de tokenização e viraria comparação de tamanho de
modelo. Vocabulário pequeno mantém o orçamento de parâmetros parecido com o char-level.

Uso:
    python3 run_tokenization_comparison.py                # comparação completa
    python3 run_tokenization_comparison.py --smoke-test    # valida o pipeline em ~1 min
"""
import argparse
import csv
import json
import math
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bpe_tokenizer import train_bpe
from kfold_split import assign_folds, build_global_vocab, load_works, write_fold_bins, write_fold_bins_bpe

HERE = Path(__file__).parent
STEP_RE = re.compile(r"step (\d+): train loss ([\d.]+), val loss ([\d.]+)")
LN2 = math.log(2)

BASE_CONFIG = dict(
    n_layer=6, n_head=6, n_embd=384, block_size=256, dropout=0.2,
    batch_size=128, learning_rate=1e-3,
)


def detect_device():
    import torch
    if torch.cuda.is_available():
        return "cuda", True
    if torch.backends.mps.is_available():
        return "mps", False
    return "cpu", False


def get_or_train_bpe(input_txt: Path, vocab_size: int, sample_chars: int, cache_path: Path) -> dict:
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if cached.get("vocab_size_target") == vocab_size and cached.get("sample_chars") == sample_chars:
            print(f"[bpe] reaproveitando tokenizer cacheado em {cache_path} (vocab={len(cached['tokenizer']['vocab'])})")
            return cached["tokenizer"]
    text = input_txt.read_text(encoding="utf-8")
    base_chars = sorted(set(text))
    print(f"[bpe] treinando BPE (vocab_size={vocab_size}, amostra={sample_chars:,} caracteres)...".replace(",", "."))
    t0 = time.time()
    tokenizer = train_bpe(text, vocab_size=vocab_size, base_chars=base_chars, sample_chars=sample_chars)
    print(f"[bpe] treinado em {time.time()-t0:.1f}s, vocab final = {len(tokenizer['vocab'])}")
    with open(cache_path, "wb") as f:
        pickle.dump({"vocab_size_target": vocab_size, "sample_chars": sample_chars, "tokenizer": tokenizer}, f)
    return tokenizer


def prepare_tokenization_folds(nanogpt_dir: Path, input_txt: Path, manifest: Path, k: int, seed: int,
                                bpe_vocab_size: int, bpe_sample_chars: int, force: bool):
    works = load_works(input_txt, manifest)
    fold_of = assign_folds(works, k, seed)

    # --- char-level (mesmo vocabulário global de sempre) ---
    stoi = build_global_vocab(input_txt)
    char_dirs = []
    for f in range(k):
        out_dir = nanogpt_dir / "data" / f"hpsearch_fold{f}"
        if force or not (out_dir / "train.bin").exists():
            n_train, n_val = write_fold_bins(works, fold_of, f, stoi, out_dir)
            print(f"[folds][char] fold {f}: {n_train:,} treino / {n_val:,} val tokens".replace(",", "."))
        char_dirs.append(out_dir)

    # --- BPE (vocabulário pequeno treinado no corpus) ---
    cache_path = HERE / f"bpe_cache_v{bpe_vocab_size}.pkl"
    tokenizer = get_or_train_bpe(input_txt, bpe_vocab_size, bpe_sample_chars, cache_path)
    bpe_dirs = []
    bpe_stats = []
    for f in range(k):
        out_dir = nanogpt_dir / "data" / f"hpsearch_fold{f}_bpe{bpe_vocab_size}"
        stats_path = out_dir / "token_stats.json"
        if force or not stats_path.exists():
            stats = write_fold_bins_bpe(works, fold_of, f, tokenizer, out_dir)
            print(f"[folds][bpe] fold {f}: {stats['train_tokens']:,} treino / {stats['val_tokens']:,} val "
                  f"tokens ({stats['val_chars']/max(stats['val_tokens'],1):.2f} chars/token)".replace(",", "."))
        else:
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
        bpe_dirs.append(out_dir)
        bpe_stats.append(stats)

    return char_dirs, bpe_dirs, bpe_stats, tokenizer


def build_config_file(dataset_name, max_iters, device, compile_flag, out_dir_name, learning_rate, warmup_iters):
    cfg = BASE_CONFIG
    eval_interval = max(50, max_iters // 4)
    eval_iters = min(50, max(10, max_iters // 20))
    return f"""
out_dir = '{out_dir_name}'
eval_interval = {eval_interval}
eval_iters = {eval_iters}
log_interval = {max(10, max_iters // 20)}
always_save_checkpoint = False
wandb_log = False

dataset = '{dataset_name}'
gradient_accumulation_steps = 1
batch_size = {cfg['batch_size']}
block_size = {cfg['block_size']}

n_layer = {cfg['n_layer']}
n_head = {cfg['n_head']}
n_embd = {cfg['n_embd']}
dropout = {cfg['dropout']}

learning_rate = {learning_rate}
max_iters = {max_iters}
lr_decay_iters = {max_iters}
min_lr = {learning_rate / 10}
beta2 = 0.99
warmup_iters = {warmup_iters}

device = '{device}'
compile = {compile_flag}
"""


def run_trial(nanogpt_dir: Path, tag: str, dataset_name: str, fold: int, max_iters: int, device: str, compile_flag: bool,
              learning_rate, warmup_iters):
    out_dir_name = f"hpsearch/out/tokcmp_{tag}_fold{fold}"
    config_text = build_config_file(dataset_name, max_iters, device, compile_flag, out_dir_name, learning_rate, warmup_iters)
    config_path = nanogpt_dir / "config" / "tokcmp_tmp.py"
    config_path.write_text(config_text, encoding="utf-8")

    log_dir = nanogpt_dir / "hpsearch" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"tokcmp_{tag}_fold{fold}.log"

    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "train.py", str(config_path.relative_to(nanogpt_dir))],
        cwd=str(nanogpt_dir), capture_output=True, text=True,
    )
    elapsed = time.time() - t0
    log_path.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")

    matches = STEP_RE.findall(proc.stdout + proc.stderr)
    if not matches:
        print(f"    [aviso] {tag} fold{fold} não produziu nenhuma linha 'step X: ...' — ver {log_path}")
        return None, None, elapsed
    _, train_loss, val_loss = matches[-1]
    return float(train_loss), float(val_loss), elapsed


def load_results(results_csv: Path):
    done = {}
    if results_csv.exists():
        with open(results_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done[(row["tokenizacao"], row["fold"])] = row
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
    ap.add_argument("--input-txt", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--bpe-vocab-size", type=int, default=1500)
    ap.add_argument("--bpe-sample-chars", type=int, default=2_000_000)
    ap.add_argument("--char-learning-rate", type=float, default=BASE_CONFIG["learning_rate"])
    ap.add_argument("--bpe-learning-rate", type=float, default=3e-4,
                     help="menor que o do char de propósito: com o LR "
                          "do char-level (1e-3), a rodada de BPE diverge (tabela de embeddings "
                          "maior, vocab_size=1500 vs 177 — mesmo grad_clip=1.0 padrão do nanoGPT "
                          "não segurou). LR 3x menor + mais warmup corrige.")
    ap.add_argument("--bpe-warmup-iters", type=int, default=300)
    ap.add_argument("--force-refold", action="store_true")
    ap.add_argument("--results-csv", default=str(HERE / "resultados_busca_tokenizacao.csv"))
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    if args.smoke_test:
        args.k = 2
        args.folds = 1
        args.iters = 20
        args.bpe_vocab_size = 300
        args.bpe_sample_chars = 200_000
        print("[smoke-test] rodando com valores mínimos só pra validar o pipeline "
              "(resultado NÃO tem valor científico)")

    nanogpt_dir = Path(args.nanogpt_dir).resolve()
    input_txt = Path(args.input_txt) if args.input_txt else nanogpt_dir / "data" / "machado_char" / "input.txt"
    manifest = Path(args.manifest) if args.manifest else nanogpt_dir / "data" / "machado_char" / "manifest.json"
    results_csv = Path(args.results_csv)

    if not input_txt.exists():
        sys.exit(f"Não encontrei {input_txt} — rode o notebook (seção do corpus) antes.")

    device, compile_flag = detect_device()
    print(f"[device] {device} (compile={compile_flag})")

    char_dirs, bpe_dirs, bpe_stats, tokenizer = prepare_tokenization_folds(
        nanogpt_dir, input_txt, manifest, args.k, args.seed,
        args.bpe_vocab_size, args.bpe_sample_chars, args.force_refold,
    )

    done = load_results(results_csv)
    if done:
        print(f"[resume] {len(done)} trials já registrados em {results_csv}, serão reaproveitados")
    write_header = not results_csv.exists()

    print(f"\nConfig fixa (base = melhor config conhecida, resultados 3): {BASE_CONFIG}")

    warmup_default = min(100, max(10, args.iters // 10))
    plan = [
        ("char", [d.name for d in char_dirs], None, args.char_learning_rate, warmup_default),
        (f"bpe{args.bpe_vocab_size}", [d.name for d in bpe_dirs], bpe_stats, args.bpe_learning_rate, args.bpe_warmup_iters),
    ]

    bpc_means = {}
    for tag, dataset_names, stats_list, lr, warmup_iters in plan:
        val_losses = []
        val_bpcs = []
        for fold in range(args.folds):
            key = (tag, str(fold))
            dataset_name = dataset_names[fold]
            if key in done:
                row = done[key]
                val_loss = float(row["val_loss"]) if row["val_loss"] else None
                val_bpc = float(row["val_bpc"]) if row["val_bpc"] else None
                print(f"  [{tag}] fold {fold}: val loss {val_loss} / BPC {val_bpc} (do CSV, pulado)")
            else:
                print(f"  [{tag}] fold {fold}: rodando ({args.iters} iters, lr={lr}, dataset={dataset_name})...")
                train_loss, val_loss, elapsed = run_trial(nanogpt_dir, tag, dataset_name, fold, args.iters, device, compile_flag,
                                                            lr, warmup_iters)
                if val_loss is None:
                    val_bpc = None
                elif stats_list is None:
                    ratio = 1.0  # char-level: 1 token == 1 caractere
                    val_bpc = (val_loss / LN2) * ratio
                else:
                    st = stats_list[fold]
                    ratio = st["val_tokens"] / st["val_chars"]
                    val_bpc = (val_loss / LN2) * ratio
                row = {
                    "tokenizacao": tag,
                    "fold": fold,
                    "max_iters": args.iters,
                    "train_loss": train_loss if train_loss is not None else "",
                    "val_loss": val_loss if val_loss is not None else "",
                    "val_bpc": round(val_bpc, 5) if val_bpc is not None else "",
                    "elapsed_s": round(elapsed, 1),
                }
                append_result(results_csv, row, write_header)
                write_header = False
                done[key] = {k: str(v) for k, v in row.items()}
                print(f"    -> val loss {val_loss}, BPC {val_bpc:.4f} ({elapsed:.0f}s)" if val_bpc is not None
                      else f"    -> falhou ({elapsed:.0f}s)")
            if val_loss is not None:
                val_losses.append(val_loss)
            if val_bpc is not None:
                val_bpcs.append(val_bpc)
        bpc_means[tag] = sum(val_bpcs) / len(val_bpcs) if val_bpcs else float("inf")

    print("\n" + "=" * 70)
    print(f"Resultado — {args.folds} fold(s), {args.iters} iterações (val loss cru NÃO é comparável")
    print("entre tokenizações — comparar só a coluna BPC):")
    for tag, _, _, _, _ in sorted(plan, key=lambda p: bpc_means[p[0]]):
        print(f"  {tag:10s} BPC médio = {bpc_means[tag]:.4f} bits/caractere")
    print("=" * 70)
    print(f"\nTodos os trials estão em: {results_csv}")


if __name__ == "__main__":
    main()
