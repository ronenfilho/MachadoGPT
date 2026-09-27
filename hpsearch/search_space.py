"""
[código próprio] Espaço de busca de hiperparâmetros (e arquitetura) para run_search.py.

Trata arquitetura (n_layer, n_head, n_embd, block_size) como hiperparâmetro comum, amostrado
junto com os de treino (dropout, learning_rate, batch_size) — não como uma etapa separada. Isso
é o que dá o "se necessário arquitetura" pedido: a busca só vai preferir um modelo maior
(n_layer/n_embd maiores) se ele realmente vencer nas rodadas curtas; senão, o "baby GPT" atual
(6/6/384) continua sendo a escolha, sem custo extra de treinar modelos maiores à toa.

Intervalos escolhidos em torno da configuração usada em `resultados 1/2/3` (6 camadas, 6
cabeças, 384 dims, block_size 256, dropout 0.2, lr 1e-3, batch_size 64→128) — a busca explora
tanto pra baixo (modelos menores/mais rápidos, caso o "baby GPT" já esteja sobre-parametrizado
pro tempo de treino curto usado nos estágios iniciais) quanto pra cima (mais capacidade, já que
`resultados 3/` mostrou a curva de val loss desacelerando perto do limite dessa arquitetura).
"""
import random

SEARCH_SPACE = {
    "n_layer": [4, 6, 8],
    "n_head": [4, 6, 8],
    "n_embd": [256, 384, 512],
    "block_size": [128, 256, 384],
    "dropout": [0.1, 0.2, 0.3],
    "learning_rate": [5e-4, 1e-3, 2e-3],
    "batch_size": [64, 128],
}


def sample_config(rng: random.Random) -> dict:
    """Amostra uma config válida (n_embd múltiplo de n_head — exigência do nanoGPT para a
    atenção multi-cabeça)."""
    n_embd = rng.choice(SEARCH_SPACE["n_embd"])
    valid_heads = [h for h in SEARCH_SPACE["n_head"] if n_embd % h == 0]
    return {
        "n_layer": rng.choice(SEARCH_SPACE["n_layer"]),
        "n_head": rng.choice(valid_heads),
        "n_embd": n_embd,
        "block_size": rng.choice(SEARCH_SPACE["block_size"]),
        "dropout": rng.choice(SEARCH_SPACE["dropout"]),
        "learning_rate": rng.choice(SEARCH_SPACE["learning_rate"]),
        "batch_size": rng.choice(SEARCH_SPACE["batch_size"]),
    }


def sample_unique_configs(n: int, seed: int) -> list[dict]:
    """Amostra `n` configs distintas (por valor, não só por seed) — evita desperdiçar um trial
    inteiro repetindo uma config já sorteada."""
    rng = random.Random(seed)
    seen = set()
    configs = []
    attempts = 0
    while len(configs) < n and attempts < n * 50:
        attempts += 1
        cfg = sample_config(rng)
        key = tuple(sorted(cfg.items()))
        if key in seen:
            continue
        seen.add(key)
        configs.append(cfg)
    return configs
