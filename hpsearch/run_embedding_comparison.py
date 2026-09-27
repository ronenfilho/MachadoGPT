#!/usr/bin/env python3
"""
[código próprio] Comparação de 4 estratégias de embedding posicional, arquitetura e demais
hiperparâmetros FIXOS na melhor config conhecida (ver `resultados 3/`). Ver
o README do projeto, seção de busca de hiperparâmetros, pro racional completo.

Diferente de `run_search.py` (que varia arquitetura+hiperparâmetros juntos), este script isola
UMA variável — a estratégia de posição — mantendo tudo mais igual, pra dar uma comparação limpa:

    'learned'    : embedding de posição aprendida (padrão GPT-2/nanoGPT) — baseline atual.
    'sinusoidal' : tabela fixa senoidal (Vaswani et al., 2017) — sem parâmetros extras.
    'rope'       : Rotary Position Embedding (Su et al., 2021) — posição dentro da atenção.
    'none'       : sem informação de posição — piso, quantifica quanto ela realmente importa.

Cada variante é avaliada nos mesmos `--folds` folds do split k-fold por obra (kfold_split.py),
com o mesmo orçamento de iterações — só o `pos_emb_type` muda entre elas.

Pré-requisito: rodar `patch_model_pos_emb.py` uma vez (idempotente — este script já chama
sozinho antes de treinar).

Uso:
    python3 run_embedding_comparison.py                # comparação completa (padrão: 3 folds, 3000 iters)
    python3 run_embedding_comparison.py --smoke-test    # valida o pipeline em ~1 min
"""
import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from kfold_split import prepare_folds
from patch_model_pos_emb import apply_patches, MODEL_PATCHES, TRAIN_PATCHES

HERE = Path(__file__).parent
STEP_RE = re.compile(r"step (\d+): train loss ([\d.]+), val loss ([\d.]+)")

# config fixa — vencedora de resultados 1/2/3
BASE_CONFIG = dict(
    n_layer=6, n_head=6, n_embd=384, block_size=256, dropout=0.2,
    batch_size=128, learning_rate=1e-3,
)
POS_EMB_TYPES = ["learned", "sinusoidal", "rope", "none"]


def detect_device():
    import torch
    if torch.cuda.is_available():
        return "cuda", True
    if torch.backends.mps.is_available():
        return "mps", False
    return "cpu", False


def build_config_file(pos_emb_type, fold, max_iters, device, compile_flag, out_dir_name):
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

dataset = 'hpsearch_fold{fold}'
gradient_accumulation_steps = 1
batch_size = {cfg['batch_size']}
block_size = {cfg['block_size']}

n_layer = {cfg['n_layer']}
n_head = {cfg['n_head']}
n_embd = {cfg['n_embd']}
dropout = {cfg['dropout']}

learning_rate = {cfg['learning_rate']}
max_iters = {max_iters}
lr_decay_iters = {max_iters}
min_lr = {cfg['learning_rate'] / 10}
beta2 = 0.99
warmup_iters = {min(100, max(10, max_iters // 10))}

device = '{device}'
compile = {compile_flag}
pos_emb_type = '{pos_emb_type}'
"""


def run_trial(nanogpt_dir: Path, pos_emb_type: str, fold: int, max_iters: int, device: str, compile_flag: bool):
    out_dir_name = f"hpsearch/out/embcmp_{pos_emb_type}_fold{fold}"
    config_text = build_config_file(pos_emb_type, fold, max_iters, device, compile_flag, out_dir_name)
    config_path = nanogpt_dir / "config" / "embcmp_tmp.py"
    config_path.write_text(config_text, encoding="utf-8")

    log_dir = nanogpt_dir / "hpsearch" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"embcmp_{pos_emb_type}_fold{fold}.log"

    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "train.py", str(config_path.relative_to(nanogpt_dir))],
        cwd=str(nanogpt_dir), capture_output=True, text=True,
    )
    elapsed = time.time() - t0
    log_path.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")

    matches = STEP_RE.findall(proc.stdout + proc.stderr)
    if not matches:
        print(f"    [aviso] {pos_emb_type} fold{fold} não produziu nenhuma linha 'step X: ...' "
              f"— ver {log_path}")
        return None, None, elapsed
    _, train_loss, val_loss = matches[-1]
    return float(train_loss), float(val_loss), elapsed


def load_results(results_csv: Path):
    done = {}
    if results_csv.exists():
        with open(results_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done[(row["pos_emb_type"], row["fold"])] = row
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
    ap.add_argument("--k", type=int, default=5, help="nº total de folds no split global (mesma seed de run_search.py reaproveita os dados)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=3, help="quantos dos k folds usar na comparação")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--results-csv", default=str(HERE / "resultados_busca_embeddings.csv"))
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    if args.smoke_test:
        args.k = 2
        args.folds = 1
        args.iters = 20
        print("[smoke-test] rodando com valores mínimos só pra validar o pipeline "
              "(resultado NÃO tem valor científico)")

    nanogpt_dir = Path(args.nanogpt_dir).resolve()
    input_txt = Path(args.input_txt) if args.input_txt else nanogpt_dir / "data" / "machado_char" / "input.txt"
    manifest = Path(args.manifest) if args.manifest else nanogpt_dir / "data" / "machado_char" / "manifest.json"
    results_csv = Path(args.results_csv)

    if not input_txt.exists():
        sys.exit(f"Não encontrei {input_txt} — rode o notebook (seção do corpus) pelo menos até "
                  "gerar data/machado_char/input.txt antes da comparação.")

    print("[patch] garantindo que model.py/train.py suportam pos_emb_type...")
    apply_patches(nanogpt_dir / "model.py", MODEL_PATCHES, "model.py")
    apply_patches(nanogpt_dir / "train.py", TRAIN_PATCHES, "train.py")

    device, compile_flag = detect_device()
    print(f"[device] {device} (compile={compile_flag})")

    prepare_folds(nanogpt_dir, input_txt, manifest, args.k, args.seed)

    done = load_results(results_csv)
    if done:
        print(f"[resume] {len(done)} trials já registrados em {results_csv}, serão reaproveitados")
    write_header = not results_csv.exists()

    print(f"\nConfig fixa (base = melhor config conhecida, resultados 3): {BASE_CONFIG}")
    means = {}
    for pos_emb_type in POS_EMB_TYPES:
        val_losses = []
        for fold in range(args.folds):
            key = (pos_emb_type, str(fold))
            if key in done:
                row = done[key]
                val_loss = float(row["val_loss"]) if row["val_loss"] else None
                print(f"  [{pos_emb_type}] fold {fold}: val loss {val_loss} (do CSV, pulado)")
            else:
                print(f"  [{pos_emb_type}] fold {fold}: rodando ({args.iters} iters)...")
                train_loss, val_loss, elapsed = run_trial(nanogpt_dir, pos_emb_type, fold, args.iters, device, compile_flag)
                row = {
                    "pos_emb_type": pos_emb_type,
                    "fold": fold,
                    "max_iters": args.iters,
                    "train_loss": train_loss if train_loss is not None else "",
                    "val_loss": val_loss if val_loss is not None else "",
                    "elapsed_s": round(elapsed, 1),
                }
                append_result(results_csv, row, write_header)
                write_header = False
                done[key] = {k: str(v) for k, v in row.items()}
                print(f"    -> val loss {val_loss} ({elapsed:.0f}s)")
            if val_loss is not None:
                val_losses.append(val_loss)
        means[pos_emb_type] = sum(val_losses) / len(val_losses) if val_losses else float("inf")

    print("\n" + "=" * 60)
    print(f"Resultado — val loss média em {args.folds} fold(s), {args.iters} iterações:")
    for pos_emb_type in sorted(POS_EMB_TYPES, key=lambda p: means[p]):
        marker = " <-- baseline" if pos_emb_type == "learned" else ""
        print(f"  {pos_emb_type:12s} {means[pos_emb_type]:.4f}{marker}")
    print("=" * 60)
    print(f"\nTodos os trials estão em: {results_csv}")


if __name__ == "__main__":
    main()
