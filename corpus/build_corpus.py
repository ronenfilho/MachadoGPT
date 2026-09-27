"""
Monta o corpus de treinamento (train.txt) a partir de DUAS fontes de domínio
público de obras de Machado de Assis:

  1) Project Gutenberg — 12 livros (romances, contos, poesia) baixados como
     .txt e limpos do boilerplate padrão de licença do Gutenberg.
  2) Wikisource em português (pt.wikisource.org) — coleção bem mais ampla:
     contos avulsos, crônicas, poesias, crítica literária, teatro, discursos
     e capítulos de romances, cobrindo praticamente toda a obra disponível
     na "Categoria:Machado de Assis" e suas ~93 subcategorias.

Autoria: script escrito para o Projeto Individual 1 (PPMEC0070 / RNAP, 2026.2).

## Sobre a parte do Wikisource

A *descoberta* de quais páginas existem (recursão por 93 subcategorias via
`action=query&list=categorymembers` da API do MediaWiki, ~1900 páginas)
já foi feita e fica **congelada** em `wikisource_pages.json` — não é refeita
a cada execução deste script (evita recrawl pesado e mantém o build
reprodutível). O que este script faz a cada execução é buscar o *texto
atual* dessas páginas e limpá-lo. `wikisource_pages.json` guarda 3 listas:

  - `direct_titles`: páginas cujo conteúdo já está inline no wikitext
    (a maioria dos contos/poemas curtos).
  - `transclusions`: páginas cujo texto real é *transcluído* de scans de
    livro no namespace `Página:` (tag `<pages index="..." from=N to=M/>`)
    — comum em romances/coletâneas que passaram pelo fluxo de revisão
    (proofreading) do Wikisource. Para essas, buscamos cada página
    `Página:<index>/<N>` do intervalo e concatenamos o texto revisado.
  - `excluded_translations`: páginas descartadas por serem traduções de
    OUTROS autores feitas por Machado (ex: "O Corvo", de Edgar Allan Poe;
    poemas de Lamartine, Schiller, poesia chinesa clássica via francês) —
    mantidas fora do corpus para não diluir o estilo autoral de Machado
    com a voz/conteúdo de outro escritor traduzido.

O limpador de wikitext (`clean_wikitext_common`) é uma solução pragmática
via regex (não um parser MediaWiki completo): remove templates `{{...}}`,
blocos `<noinclude>` (metadados de revisão), tags `<poem>`/HTML, links
`[[...]]` (mantendo o texto de exibição), categorias, hifens suaves de
OCR, e decodifica entidades HTML. Foi validado manualmente numa amostra
diversificada de páginas antes de rodar no conjunto inteiro.

Uso:
    python3 build_corpus.py
Gera:
    corpus/train.txt          (todas as obras concatenadas)
    corpus/manifest.json      (proveniência: título, fonte, nº de caracteres)
"""
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
RAW_DIR = HERE / "raw_gutenberg"
WS_PAGES_FILE = HERE / "wikisource_pages.json"
OUT_TXT = HERE / "train.txt"
OUT_MANIFEST = HERE / "manifest.json"

# ---------------------------------------------------------------------------
# Project Gutenberg
# ---------------------------------------------------------------------------

GUTENBERG_START_RE = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL)
GUTENBERG_END_RE = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*", re.IGNORECASE | re.DOTALL)

# título legível : nome do arquivo baixado
GUTENBERG_SOURCES = {
    "Dom Casmurro": "Dom_Casmurro_pg55752.txt",
    "Memórias Póstumas de Brás Cubas": "Memorias_Postumas_de_Bras_Cubas_pg54829.txt",
    "Quincas Borba": "Quincas_Borba_pg55682.txt",
    "Papéis Avulsos": "Papeis_Avulsos_pg57001.txt",
    "Poesias Completas": "Poesias_Completas_pg61653.txt",
    "A Mão e a Luva": "A_Mao_e_a_Luva_pg53101.txt",
    "Histórias sem Data": "Historias_sem_Data_pg33056.txt",
    "Helena": "Helena_pg67162.txt",
    "Iaiá Garcia": "Iaia_Garcia_pg67780.txt",
    "Esaú e Jacó": "Esau_e_Jacob_pg56737.txt",
    "Relíquias de Casa Velha": "Reliquias_de_Casa_Velha_pg67935.txt",
    "Memorial de Aires": "Memorial_de_Aires_pg55797.txt",
}


def strip_gutenberg_boilerplate(raw: str) -> str:
    """Remove cabeçalho/rodapé padrão do Project Gutenberg, mantendo só a obra."""
    m_start = GUTENBERG_START_RE.search(raw)
    body = raw[m_start.end():] if m_start else raw
    m_end = GUTENBERG_END_RE.search(body)
    body = body[: m_end.start()] if m_end else body
    return body.strip("\n") + "\n"


def build_gutenberg_corpus():
    chunks = []
    manifest = []
    for title, filename in GUTENBERG_SOURCES.items():
        path = RAW_DIR / filename
        raw = path.read_text(encoding="utf-8", errors="ignore")
        body = strip_gutenberg_boilerplate(raw)
        chunks.append(f"\n\n===== {title} (Machado de Assis) =====\n\n" + body)
        manifest.append(
            {
                "titulo": title,
                "fonte": "Project Gutenberg (dominio publico)",
                "url": f"https://www.gutenberg.org/ebooks/{filename.split('pg')[-1].split('.')[0]}",
                "n_caracteres": len(body),
            }
        )
    return chunks, manifest


# ---------------------------------------------------------------------------
# Wikisource (pt.wikisource.org)
# ---------------------------------------------------------------------------

WIKI_API = "https://pt.wikisource.org/w/api.php"
WIKI_HEADERS = {
    "User-Agent": "PPMEC0070-Projeto1-MachadoGPT/1.0 (uso academico pessoal; contato: ronen.filho@gmail.com)"
}

NAVEGAR_RE = re.compile(r"\{\{navegar(.*?)\n\}\}", re.S)
TEMPLATE_RE = re.compile(r"\{\{.*?\}\}", re.S)
DATELINE_TEMPLATE_RE = re.compile(r"\{\{[dD]\|(.*?)\}\}", re.S)
NOINCLUDE_RE = re.compile(r"<noinclude>.*?</noinclude>", re.S)
POEM_TAG_RE = re.compile(r"</?poem[^>]*>")
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
HTML_TAG_RE = re.compile(r"<[^>]+>")
CATEGORY_LINK_RE = re.compile(r"\[\[Categoria:.*?\]\]")
INTERWIKI_LINK_RE = re.compile(r"\[\[[a-z]{2,3}:.*?\]\]")
PIPED_LINK_RE = re.compile(r"\[\[([^\]|]*)\|([^\]]*)\]\]")
SIMPLE_LINK_RE = re.compile(r"\[\[([^\]]*)\]\]")
SOFT_HYPHEN = "­"


def _wiki_api_get(params: dict) -> dict:
    url = WIKI_API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=WIKI_HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _fetch_wikitext_batch(titles: list[str]) -> dict[str, str | None]:
    d = _wiki_api_get(
        {
            "action": "query",
            "prop": "revisions",
            "rvprop": "content",
            "rvslots": "main",
            "titles": "|".join(titles),
            "format": "json",
        }
    )
    pages = d.get("query", {}).get("pages", {})
    out = {}
    for p in pages.values():
        revs = p.get("revisions", [])
        out[p.get("title")] = revs[0]["slots"]["main"]["*"] if revs else None
    return out


def fetch_wikitext_many(titles: list[str], batch_size: int = 50, delay: float = 0.2, retries: int = 3) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for i in range(0, len(titles), batch_size):
        batch = titles[i : i + batch_size]
        for attempt in range(retries):
            try:
                result.update(_fetch_wikitext_batch(batch))
                break
            except urllib.error.URLError:
                if attempt == retries - 1:
                    for t in batch:
                        result.setdefault(t, None)
                else:
                    time.sleep(1.0)
        time.sleep(delay)
    return result


def clean_wikitext_common(t: str) -> str:
    """Limpeza de wikitext -> texto puro (regex-based, ver docstring do módulo)."""
    t = NOINCLUDE_RE.sub("", t)
    t = re.sub(r"<nowiki\s*/?>", "", t)
    t = re.sub(r"<references\s*/?>", "", t)
    t = HTML_COMMENT_RE.sub("", t)
    t = POEM_TAG_RE.sub("", t)
    t = t.replace(SOFT_HYPHEN, "")
    t = CATEGORY_LINK_RE.sub("", t)
    t = INTERWIKI_LINK_RE.sub("", t)
    t = DATELINE_TEMPLATE_RE.sub(r"\1", t)  # {{d|texto}} (dateline/assinatura de carta) -> texto
    for _ in range(3):  # múltiplas passadas: templates às vezes ficam aninhados de forma simples
        new_t = TEMPLATE_RE.sub("", t)
        if new_t == t:
            break
        t = new_t
    t = PIPED_LINK_RE.sub(lambda m: m.group(2), t)
    t = SIMPLE_LINK_RE.sub(lambda m: m.group(1), t)
    t = HTML_TAG_RE.sub("", t)
    t = t.replace("'''", "").replace("''", "")
    t = html.unescape(t)
    t = html.unescape(t)  # entidades duplamente escapadas (ex: "&amp;amp;") de algumas fontes OCR
    t = re.sub(r"<br\s*/?>", "\n", t)
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    # rede de segurança final: qualquer chave/colchete duplo remanescente aqui só pode ser
    # resíduo de template/link mal-formado ou aninhado que escapou das passadas acima
    # (ex: {{a|{{b}}}} aninhado) — nesse ponto não é mais recuperável como estrutura, então
    # é só ruído a remover.
    t = re.sub(r"\{\{|\}\}|\[\[|\]\]", "", t)
    return t.strip()


def clean_direct_page(raw: str) -> str:
    t = NAVEGAR_RE.sub("", raw, count=1)
    return clean_wikitext_common(t)


def build_wikisource_corpus():
    ws_pages = json.loads(WS_PAGES_FILE.read_text(encoding="utf-8"))
    direct_titles: list[str] = ws_pages["direct_titles"]
    transclusions: dict[str, dict] = ws_pages["transclusions"]

    print(f"Wikisource: buscando {len(direct_titles)} paginas diretas...")
    direct_raw = fetch_wikitext_many(direct_titles)

    page_ns_titles = sorted(
        {f"Página:{info['index']}/{n}" for info in transclusions.values() for n in info["pages"]}
    )
    print(f"Wikisource: buscando {len(page_ns_titles)} paginas de scan (namespace Página:)...")
    page_ns_raw = fetch_wikitext_many(page_ns_titles)

    chunks = []
    manifest = []

    for title in direct_titles:
        raw = direct_raw.get(title)
        if not raw:
            continue
        text = clean_direct_page(raw)
        if len(text) < 20:
            continue
        chunks.append(f"\n\n===== {title} (Machado de Assis, Wikisource) =====\n\n" + text)
        manifest.append({"titulo": title, "fonte": "Wikisource (dominio publico)", "n_caracteres": len(text)})

    for title, info in transclusions.items():
        parts = []
        for n in info["pages"]:
            raw = page_ns_raw.get(f"Página:{info['index']}/{n}")
            if raw:
                cleaned = clean_wikitext_common(raw)
                if cleaned:
                    parts.append(cleaned)
        text = re.sub(r"\n{3,}", "\n\n", " ".join(parts)).strip()
        if len(text) < 20:
            continue
        chunks.append(f"\n\n===== {title} (Machado de Assis, Wikisource) =====\n\n" + text)
        manifest.append({"titulo": title, "fonte": "Wikisource (dominio publico)", "n_caracteres": len(text)})

    return chunks, manifest


# ---------------------------------------------------------------------------


def main():
    gutenberg_chunks, gutenberg_manifest = build_gutenberg_corpus()
    ws_chunks, ws_manifest = build_wikisource_corpus()

    all_chunks = gutenberg_chunks + ws_chunks
    all_manifest = gutenberg_manifest + ws_manifest

    full_text = "".join(all_chunks)
    OUT_TXT.write_text(full_text, encoding="utf-8")
    OUT_MANIFEST.write_text(json.dumps(all_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    total_chars = len(full_text)
    total_words = len(full_text.split())
    print(f"\nCorpus gerado: {OUT_TXT}")
    print(f"  Gutenberg: {len(gutenberg_manifest)} obras")
    print(f"  Wikisource: {len(ws_manifest)} paginas")
    print(f"  Total: {total_chars:,} caracteres, {total_words:,} palavras".replace(",", "."))


if __name__ == "__main__":
    main()
