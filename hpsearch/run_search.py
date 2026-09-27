#!/usr/bin/env python3
"""
[código próprio] Busca de hiperparâmetros/arquitetura por successive halving + k-fold por obra,
pro Projeto Individual 1 (PPMEC0070). Ver a seção "Busca de
hiperparâmetros", para a estratégia completa e o racional de cada escolha.

Resumo do algoritmo:
  1. Um k-fold GLOBAL por obra é montado uma única vez (kfold_split.py) e cacheado em
     nanoGPT/data/hpsearch_fold{0..k-1}/ — mesmo vocabulário fixo em todos os folds.
  2. Successive halving em 3 estágios, cada um mais caro (mais iterações, mais folds) e com
     menos configs sobrevivendo — a maioria das configs é descartada cedo, barata:
        estágio 1: N configs   × 1 fold  × poucas iterações   (triagem)
        estágio 2: N/2 configs × 2 folds × mais iterações      (confirmação)
        estágio 3: N/4 configs × 3 folds × ainda mais iterações (decisão final)
  3. Config vencedora = menor val loss média no estágio 3. Recomendação de próximo passo
     (rodada final de orçamento completo, ex: 15000 iterações, split 90/10 padrão) é impressa
     no final, não executada automaticamente por este script (fica pra um `resultados 4/`
     manual, revisado antes de rodar).

Uso típico (a partir de Projeto-1/hpsearch/):
    python3 run_search.py                      # roda a busca completa (padrão: k=5)
    python3 run_search.py --smoke-test         # valida o pipeline em ~1-2 min (iters mínimas)
    python3 run_search.py --resume             # continua de um resultados_busca.csv existente

É resumível: cada trial (estágio, config, fold) já registrado em `--results-csv` é pulado.
Interromper com Ctrl-C a qualquer momento é seguro — o CSV é escrito trial a trial.
"""
import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from kfold_split import prepare_folds
from search_space import sample_unique_configs

HERE = Path(__file__).parent
STEP_RE = re.compile(r"step (\d+): train loss ([\d.]+), val loss ([\d.]+)")


def detect_device():
    import torch

    if torch.cuda.is_available():
        return "cuda", True
    if torch.backends.mps.is_available():
        return "mps", False
    return "cpu", False


def config_hash(cfg: dict) -> str:
    s = json.dumps(cfg, sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:10]


def build_config_file(cfg: dict, fold: int, max_iters: int, device: str, compile_flag: bool, out_dir_name: str) -> str:
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
"""


def run_trial(nanogpt_dir: Path, cfg: dict, fold: int, max_iters: int, device: str, compile_flag: bool, tag: str):
    cid = config_hash(cfg)
    out_dir_name = f"hpsearch/out/{tag}_{cid}_fold{fold}"
    config_text = build_config_file(cfg, fold, max_iters, device, compile_flag, out_dir_name)
    config_path = nanogpt_dir / "config" / "hpsearch_tmp.py"
    config_path.write_text(config_text, encoding="utf-8")

    log_dir = nanogpt_dir / "hpsearch" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{tag}_{cid}_fold{fold}.log"

    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "train.py", str(config_path.relative_to(nanogpt_dir))],
        cwd=str(nanogpt_dir),
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0
    log_path.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")

    matches = STEP_RE.findall(proc.stdout + proc.stderr)
    if not matches:
        print(f"    [aviso] trial {cid} fold{fold} não produziu nenhuma linha 'step X: ...' "
              f"— ver {log_path}")
        return None, None, elapsed
    last_iter, train_loss, val_loss = matches[-1]
    return float(train_loss), float(val_loss), elapsed


def load_results(results_csv: Path):
    done = {}
    if results_csv.exists():
        with open(results_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row["stage"], row["config_hash"], row["fold"])
                done[key] = row
    return done


def append_result(results_csv: Path, row: dict, write_header: bool):
    with open(results_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def evaluate_stage(nanogpt_dir, configs, n_folds, max_iters, device, compile_flag, tag, results_csv, done):
    """Roda (ou reaproveita do CSV) todos os trials da etapa; retorna {config_hash: mean_val_loss}."""
    write_header = not results_csv.exists()
    means = {}
    for cfg in configs:
        cid = config_hash(cfg)
        val_losses = []
        for fold in range(n_folds):
            key = (tag, cid, str(fold))
            if key in done:
                row = done[key]
                val_loss = float(row["val_loss"]) if row["val_loss"] != "" else None
                print(f"  [{tag}] config {cid} fold {fold}: val loss {val_loss} (do CSV, pulado)")
            else:
                print(f"  [{tag}] config {cid} fold {fold}: rodando ({max_iters} iters)...")
                train_loss, val_loss, elapsed = run_trial(nanogpt_dir, cfg, fold, max_iters, device, compile_flag, tag)
                row = {
                    "stage": tag,
                    "config_hash": cid,
                    "config_json": json.dumps(cfg, sort_keys=True),
                    "fold": fold,
                    "max_iters": max_iters,
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
        means[cid] = sum(val_losses) / len(val_losses) if val_losses else float("inf")
    return means


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nanogpt-dir", default=str(HERE.parent / "notebook" / "nanoGPT"))
    ap.add_argument("--input-txt", default=None, help="default: <nanogpt-dir>/data/machado_char/input.txt")
    ap.add_argument("--manifest", default=None, help="default: <nanogpt-dir>/data/machado_char/manifest.json")
    ap.add_argument("--k", type=int, default=5, help="número total de folds no split global")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force-refold", action="store_true")

    ap.add_argument("--stage1-n", type=int, default=8)
    ap.add_argument("--stage1-iters", type=int, default=600)
    ap.add_argument("--stage1-folds", type=int, default=1)

    ap.add_argument("--stage2-iters", type=int, default=1500)
    ap.add_argument("--stage2-folds", type=int, default=2)

    ap.add_argument("--stage3-iters", type=int, default=3000)
    ap.add_argument("--stage3-folds", type=int, default=3)

    ap.add_argument("--promote-frac", type=float, default=0.5)
    ap.add_argument("--results-csv", default=str(HERE / "resultados_busca.csv"))
    ap.add_argument("--smoke-test", action="store_true",
                     help="sobrescreve tudo pra rodar em ~1-2 min e validar o pipeline, sem valor científico")

    args = ap.parse_args()

    if args.smoke_test:
        args.k = 2
        args.stage1_n = 2
        args.stage1_iters = args.stage2_iters = args.stage3_iters = 20
        args.stage1_folds = args.stage2_folds = args.stage3_folds = 1
        print("[smoke-test] rodando com valores mínimos só pra validar o pipeline "
              "(resultado NÃO tem valor científico)")

    nanogpt_dir = Path(args.nanogpt_dir).resolve()
    input_txt = Path(args.input_txt) if args.input_txt else nanogpt_dir / "data" / "machado_char" / "input.txt"
    manifest = Path(args.manifest) if args.manifest else nanogpt_dir / "data" / "machado_char" / "manifest.json"
    results_csv = Path(args.results_csv)

    if not input_txt.exists():
        sys.exit(f"Não encontrei {input_txt} — rode o notebook (seção do corpus) pelo menos até "
                  "gerar data/machado_char/input.txt antes da busca.")

    device, compile_flag = detect_device()
    print(f"[device] {device} (compile={compile_flag})")

    prepare_folds(nanogpt_dir, input_txt, manifest, args.k, args.seed, args.force_refold)

    done = load_results(results_csv)
    if done:
        print(f"[resume] {len(done)} trials já registrados em {results_csv}, serão reaproveitados")

    configs = sample_unique_configs(args.stage1_n, args.seed)
    print(f"\n=== estágio 1: {len(configs)} configs, {args.stage1_folds} fold(s), "
          f"{args.stage1_iters} iterações ===")
    means1 = evaluate_stage(nanogpt_dir, configs, args.stage1_folds, args.stage1_iters,
                             device, compile_flag, "stage1", results_csv, done)

    n2 = max(1, round(len(configs) * args.promote_frac))
    survivors2 = sorted(configs, key=lambda c: means1[config_hash(c)])[:n2]
    print(f"\n=== estágio 2: {len(survivors2)} configs promovidas, {args.stage2_folds} fold(s), "
          f"{args.stage2_iters} iterações ===")
    means2 = evaluate_stage(nanogpt_dir, survivors2, args.stage2_folds, args.stage2_iters,
                             device, compile_flag, "stage2", results_csv, done)

    n3 = max(1, round(len(survivors2) * args.promote_frac))
    survivors3 = sorted(survivors2, key=lambda c: means2[config_hash(c)])[:n3]
    print(f"\n=== estágio 3: {len(survivors3)} configs promovidas, {args.stage3_folds} fold(s), "
          f"{args.stage3_iters} iterações ===")
    means3 = evaluate_stage(nanogpt_dir, survivors3, args.stage3_folds, args.stage3_iters,
                             device, compile_flag, "stage3", results_csv, done)

    winner = min(survivors3, key=lambda c: means3[config_hash(c)])
    print("\n" + "=" * 60)
    print("Config vencedora (menor val loss média no estágio 3):")
    print(json.dumps(winner, indent=2, sort_keys=True))
    print(f"val loss média (estágio 3, {args.stage3_iters} iters, {args.stage3_folds} folds): "
          f"{means3[config_hash(winner)]:.4f}")
    print("=" * 60)
    print("\nPróximo passo (não automático): rodar essa config em orçamento completo "
          "(ex: 15000 iterações, split 90/10 padrão de data/machado_char/) e revisar "
          "antes de consolidar como uma nova pasta de resultados.")
    print(f"\nTodos os trials (inclusive descartados) estão em: {results_csv}")


if __name__ == "__main__":
    main()
