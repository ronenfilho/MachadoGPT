"""
[código próprio] Divisão do corpus em k folds por OBRA (não por caractere) — usado pela busca
de hiperparâmetros em run_search.py.

Por que por obra e não por caractere: o split 90/10 padrão do notebook (`data/machado_char/
input.txt`, corte contíguo nos últimos 10% dos caracteres) é simples e adequado pro resultado
final, mas ruim para validar hiperparâmetros — um corte contíguo joga inteiro na validação só
o fim do corpus (hoje, só páginas do Wikisource, já que Gutenberg vem primeiro na concatenação),
enquanto k-fold por caractere aleatório vazaria contexto adjacente entre treino e validação
(blocos vizinhos de um mesmo livro, quase idênticos, um de cada lado do split). Dividir por
obra inteira evita os dois problemas: cada fold de validação é um conjunto de livros/páginas que
o modelo nunca viu, nenhuma obra é cortada ao meio entre treino e validação.

Cada obra é recuperada de `data/machado_char/input.txt` usando o mesmo delimitador que
`corpus/build_corpus.py` escreve entre obras (`\\n\\n===== <titulo> =====\\n\\n`) — a ORDEM das
obras no texto é idêntica à ordem do `manifest.json` (mesmo código que gera os dois), então
basta parear por posição; não dependemos do texto exato do delimitador.
"""
import heapq
import json
import pickle
import random
import re
from pathlib import Path

import numpy as np

DELIM_RE = re.compile(r"\n\n===== .+? =====\n\n")


def load_works(input_txt_path, manifest_path):
    """Retorna lista de {titulo, fonte, text} — uma entrada por obra, na ordem do manifest."""
    text = Path(input_txt_path).read_text(encoding="utf-8")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

    matches = list(DELIM_RE.finditer(text))
    if len(matches) != len(manifest):
        raise ValueError(
            f"Número de delimitadores de obra no texto ({len(matches)}) não bate com o "
            f"manifest ({len(manifest)}) — o formato de `build_corpus.py` pode ter mudado. "
            "Reveja DELIM_RE antes de confiar no split por obra."
        )

    works = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        works.append(
            {
                "titulo": manifest[i]["titulo"],
                "fonte": manifest[i]["fonte"],
                "text": text[start:end],
            }
        )
    return works


def assign_folds(works, k, seed=0):
    """Bin-packing guloso (LPT — longest processing time first): maior obra primeiro, sempre pro
    fold com menor total acumulado. Dá folds balanceados em nº de caracteres apesar da enorme
    variação de tamanho entre obras (romance inteiro vs. página avulsa de Wikisource).
    `seed` só desempata obras de tamanho igual (o algoritmo em si é determinístico)."""
    rng = random.Random(seed)
    order = list(range(len(works)))
    rng.shuffle(order)
    order.sort(key=lambda i: len(works[i]["text"]), reverse=True)

    heap = [(0, f) for f in range(k)]
    heapq.heapify(heap)
    fold_of = [None] * len(works)
    for i in order:
        total, f = heapq.heappop(heap)
        fold_of[i] = f
        heapq.heappush(heap, (total + len(works[i]["text"]), f))
    return fold_of


def build_global_vocab(input_txt_path):
    """Vocabulário fixo, construído uma única vez a partir do corpus INTEIRO — usado em todos os
    folds/trials pra manter vocab_size (e portanto a forma da tabela de embeddings) idêntico entre
    execuções, e pra garantir que nenhum caractere de um fold de validação fique fora do vocabulário
    de treino."""
    text = Path(input_txt_path).read_text(encoding="utf-8")
    chars = sorted(set(text))
    return {ch: i for i, ch in enumerate(chars)}


def write_fold_bins(works, fold_of, val_fold, stoi, out_dir):
    """Escreve train.bin/val.bin/meta.pkl para um split treino=todos os folds exceto `val_fold`,
    validação=`val_fold`. Mesmo formato binário (uint16) e mesmo meta.pkl que o notebook usa,
    então train.py do nanoGPT consome sem nenhuma mudança."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_text = "".join(w["text"] for w, f in zip(works, fold_of) if f != val_fold)
    val_text = "".join(w["text"] for w, f in zip(works, fold_of) if f == val_fold)

    def encode(s):
        # na prática nenhum caractere deveria cair fora do vocabulário global (ele vem do corpus
        # inteiro) — o filtro é só uma rede de segurança
        return [stoi[c] for c in s if c in stoi]

    train_ids = np.array(encode(train_text), dtype=np.uint16)
    val_ids = np.array(encode(val_text), dtype=np.uint16)
    train_ids.tofile(out_dir / "train.bin")
    val_ids.tofile(out_dir / "val.bin")

    itos = {i: ch for ch, i in stoi.items()}
    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump({"vocab_size": len(stoi), "itos": itos, "stoi": stoi}, f)

    return len(train_ids), len(val_ids)


def write_fold_bins_bpe(works, fold_of, val_fold, tokenizer, out_dir):
    """Igual a `write_fold_bins`, mas codificando com um tokenizer BPE (`bpe_tokenizer.py`) em
    vez do mapa char->int. Usado por `run_tokenization_comparison.py`. Retorna também as
    contagens de CARACTERES (não só tokens) de cada split — necessário pra converter loss em
    bits-por-caractere (BPC) depois, já que o nº de tokens não é comparável entre tokenizações
    de granularidade diferente."""
    from bpe_tokenizer import encode as bpe_encode

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_text = "".join(w["text"] for w, f in zip(works, fold_of) if f != val_fold)
    val_text = "".join(w["text"] for w, f in zip(works, fold_of) if f == val_fold)

    train_ids = np.array(bpe_encode(train_text, tokenizer), dtype=np.uint16)
    val_ids = np.array(bpe_encode(val_text, tokenizer), dtype=np.uint16)
    train_ids.tofile(out_dir / "train.bin")
    val_ids.tofile(out_dir / "val.bin")

    # train.py só lê meta['vocab_size'] — não precisamos do stoi/itos completo aqui, essa pasta
    # não é usada por sample.py, só pra comparar val loss/BPC
    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump({"vocab_size": len(tokenizer["vocab"])}, f)

    stats = {
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
        "train_chars": len(train_text),
        "val_chars": len(val_text),
    }
    with open(out_dir / "token_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return stats


def fold_sizes_report(works, fold_of, k):
    sizes = [0] * k
    for w, f in zip(works, fold_of):
        sizes[f] += len(w["text"])
    return sizes


def prepare_folds(nanogpt_dir, input_txt, manifest, k, seed, force=False):
    """Monta (ou reaproveita do disco) o split k-fold completo — usado tanto por run_search.py
    quanto por run_embedding_comparison.py, pra garantir que os dois usam exatamente os mesmos
    folds (mesma seed, mesmo `k`) e não retokenizam o corpus duas vezes à toa."""
    works = load_works(input_txt, manifest)
    fold_of = assign_folds(works, k, seed)
    sizes = fold_sizes_report(works, fold_of, k)
    total = sum(sizes)
    print(f"[folds] {len(works)} obras -> {k} folds (tamanhos em % do corpus): "
          + ", ".join(f"f{f}={s/total:.1%}" for f, s in enumerate(sizes)))

    stoi = build_global_vocab(input_txt)
    print(f"[folds] vocabulário global fixo: {len(stoi)} caracteres")

    fold_dirs = []
    for f in range(k):
        out_dir = Path(nanogpt_dir) / "data" / f"hpsearch_fold{f}"
        if force or not (out_dir / "train.bin").exists():
            n_train, n_val = write_fold_bins(works, fold_of, f, stoi, out_dir)
            print(f"[folds] fold {f}: {n_train:,} tokens treino / {n_val:,} tokens val -> {out_dir}"
                  .replace(",", "."))
        else:
            print(f"[folds] fold {f}: já existe em {out_dir} (use force=True pra regerar)")
        fold_dirs.append(out_dir)
    return fold_dirs
