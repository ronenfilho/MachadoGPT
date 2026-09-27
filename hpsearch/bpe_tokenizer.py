"""
[código próprio] BPE (Byte-Pair Encoding) mínimo, treinado no próprio corpus de Machado —
usado por `run_tokenization_comparison.py` pra comparar char-level (atual) vs. subword.

Por que um BPE próprio em vez do `tiktoken`/GPT-2 (já usado em `sample.py`): o vocabulário do
GPT-2 tem 50257 tokens — pra um modelo de ~10,7M parâmetros, só a tabela de embeddings
(vocab_size × n_embd) explodiria pra ~19,3M parâmetros, maior que o resto do modelo inteiro.
Isso deixaria de ser uma comparação de TOKENIZAÇÃO e viraria uma comparação de TAMANHO DE
MODELO. Treinar um BPE pequeno (poucos milhares de tokens) no próprio corpus mantém o
orçamento de parâmetros comparável ao char-level, isolando a variável de interesse.

Algoritmo: BPE clássico (Sennrich et al., 2016, "Neural Machine Translation of Rare Words with
Subword Units") — implementação incremental (só reprocessa palavras afetadas por cada merge, não
recalcula as contagens de pares do zero a cada iteração), o suficiente pra treinar em poucos
minutos mesmo em Python puro. Vocabulário inicial = alfabeto global de caracteres (mesmo usado
pelo char-level, `kfold_split.build_global_vocab`), garantindo que nenhum caractere do corpus
fique fora do vocabulário do BPE (só os merges são aprendidos por cima disso).
"""
import re
from collections import Counter, defaultdict

WORD_RE = re.compile(r"\S+|\s+")


def _word_freqs(text: str) -> Counter:
    return Counter(WORD_RE.findall(text))


def train_bpe(text: str, vocab_size: int, base_chars: list[str], sample_chars: int = 2_000_000, verbose: bool = True) -> dict:
    """Treina merges de BPE numa AMOSTRA do corpus (prática padrão — as estatísticas de par mais
    frequente se estabilizam bem antes de ver o corpus inteiro; o próprio GPT-2 original também
    treinou o BPE numa amostra). `base_chars` é o alfabeto completo do corpus INTEIRO (não só da
    amostra) — garante que caracteres raros que não aparecem na amostra ainda fiquem no
    vocabulário como tokens de 1 caractere."""
    sample = text[:sample_chars] if sample_chars else text
    word_freqs = _word_freqs(sample)
    splits = {w: list(w) for w in word_freqs}

    pair_counts = Counter()
    pair_to_words = defaultdict(set)
    for w, freq in word_freqs.items():
        symbols = splits[w]
        for i in range(len(symbols) - 1):
            p = (symbols[i], symbols[i + 1])
            pair_counts[p] += freq
            pair_to_words[p].add(w)

    vocab = list(dict.fromkeys(base_chars))  # preserva ordem, remove duplicatas
    merges: list[tuple[str, str]] = []

    n_base = len(vocab)
    while len(vocab) < vocab_size and pair_counts:
        best_pair, best_count = pair_counts.most_common(1)[0]
        if best_count < 2:
            break
        new_symbol = "".join(best_pair)
        vocab.append(new_symbol)
        merges.append(best_pair)

        affected = list(pair_to_words[best_pair])
        del pair_counts[best_pair]
        del pair_to_words[best_pair]
        for w in affected:
            freq = word_freqs[w]
            symbols = splits[w]
            for i in range(len(symbols) - 1):
                p = (symbols[i], symbols[i + 1])
                pair_counts[p] -= freq
                if pair_counts[p] <= 0:
                    del pair_counts[p]
                pair_to_words[p].discard(w)
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == best_pair[0] and symbols[i + 1] == best_pair[1]:
                    new_symbols.append(new_symbol)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            splits[w] = new_symbols
            for i in range(len(new_symbols) - 1):
                p = (new_symbols[i], new_symbols[i + 1])
                pair_counts[p] += freq
                pair_to_words[p].add(w)

        if verbose and (len(vocab) - n_base) % 200 == 0:
            print(f"[bpe] vocab={len(vocab)} último merge={best_pair!r} freq={best_count}")

    stoi = {s: i for i, s in enumerate(vocab)}
    merge_ranks = {pair: i for i, pair in enumerate(merges)}
    return {"vocab": vocab, "merges": merges, "stoi": stoi, "merge_ranks": merge_ranks}


def _encode_word(word: str, merge_ranks: dict) -> list[str]:
    symbols = list(word)
    while len(symbols) > 1:
        candidates = [
            (merge_ranks[(symbols[i], symbols[i + 1])], i)
            for i in range(len(symbols) - 1)
            if (symbols[i], symbols[i + 1]) in merge_ranks
        ]
        if not candidates:
            break
        _, i = min(candidates)
        symbols = symbols[:i] + ["".join(symbols[i : i + 2])] + symbols[i + 2 :]
    return symbols


def encode(text: str, tokenizer: dict) -> list[int]:
    stoi = tokenizer["stoi"]
    merge_ranks = tokenizer["merge_ranks"]
    ids = []
    for word in WORD_RE.findall(text):
        for sym in _encode_word(word, merge_ranks):
            if sym in stoi:
                ids.append(stoi[sym])
            else:
                # não deveria acontecer (vocab inclui todo o alfabeto base) — rede de segurança
                for ch in sym:
                    ids.append(stoi[ch])
    return ids


def decode(ids: list[int], tokenizer: dict) -> str:
    vocab = tokenizer["vocab"]
    return "".join(vocab[i] for i in ids)
