# MachadoGPT

Um GPT-2 em nível de caractere, treinado do zero na obra completa de Machado de Assis.

Projeto Individual 1 da disciplina **PPMEC0070 — Cibernética e Aprendizagem de Máquinas** (Redes Neurais e Aprendizado Profundo), PPMEC/Universidade de Brasília, 2026/2 — Prof. Dibio Leandro Borges.

**Autor:** Ronen Rodrigues Silva Filho (matrícula 262122402)

---

## O que é

Implementação, treinamento e avaliação de um modelo de linguagem autorregressivo semelhante ao GPT-2, usando o [nanoGPT](https://github.com/karpathy/nanoGPT) de Andrej Karpathy como código base (exigência do enunciado), treinado exclusivamente em um corpus próprio com a obra de Machado de Assis.

O projeto cobre, além do treinamento em si:

- construção de um corpus de **13.978.947 caracteres** combinando Project Gutenberg (12 obras) e Wikisource em português (1.781 páginas), por um *pipeline* próprio de coleta e limpeza — publicado como *dataset* independente no HuggingFace (ver abaixo);
- três rodadas de treino controladas isolando o efeito do tamanho do corpus e do orçamento de iterações;
- uma infraestrutura de busca automatizada de arquitetura, *embedding* posicional e tokenização (validação cruzada *k-fold* por obra);
- um artigo científico em formato IEEE (dupla coluna, 6 páginas) reportando tudo isso, incluindo um achado metodológico sobre os limites do treino em precisão mista no backend MPS da Apple.

**Dataset publicado:** https://huggingface.co/datasets/ronenfilho/machado-de-assis-corpus

## Resultado principal

| Rodada | Corpus | Iterações | Val. loss | Perplexidade |
|---|---|---|---|---|
| Baseline oficial do nanoGPT (`shakespeare_char`) | ≈1MB | 5.000 | 1,4697 | 4,35 |
| `resultados 1/` — só Gutenberg | 4,0M car. | 5.000 | 1,3345 | 3,80 |
| `resultados 2/` — corpus completo | 14,0M car. | 5.000 | 1,2097 | 3,35 |
| **`resultados 3/` — corpus completo** | **14,0M car.** | **15.000** | **1,1040** | **3,02** |
| `resultados 4/` — busca de melhorias (arquitetura/RoPE/BPE) | 14,0M car. | 15.000 (por candidato) | falhou na confirmação¹ | falhou na confirmação¹ |

`resultados 3/` é o melhor resultado validado do projeto: sem sinal de *overfitting* em nenhum momento do treino.

¹ `resultados 4/` documenta uma busca automatizada por melhorias adicionais (arquitetura, RoPE, tokenização BPE): os 3 candidatos batiam a config atual num orçamento curto (validação k-fold), mas nenhum se confirmou em orçamento completo — colapsam pra um patamar pior ou divergem (`NaN`), provavelmente por falta de proteção numérica (`GradScaler`) no backend MPS. Por isso a tabela não tem um val loss/perplexidade comparável nessa linha — ver `resultados 4/README.md` e o artigo para a discussão completa.

## Estrutura do repositório

```
notebook/            notebook Google Colab autocontido (o artefato principal)
corpus/               pipeline de construção do corpus (build_corpus.py) + proveniência
                      (versão pronta publicada em huggingface.co/datasets/ronenfilho/machado-de-assis-corpus)
hpsearch/             infraestrutura de busca automatizada (k-fold, successive halving)
resultados 1/2/3/4/   resultados de cada rodada de treino (README, gráficos, amostras)
artigo/               artigo científico em LaTeX (IEEE), PDF compilado e figuras
```

## O notebook

[`notebook/projeto1_machado_gpt_Ronen_R_S_Filho_colab.ipynb`](notebook/projeto1_machado_gpt_Ronen_R_S_Filho_colab.ipynb) é autocontido: baixa e monta o corpus, treina e avalia o modelo do zero, sem depender de nenhum arquivo externo deste repositório. Roda tanto no Google Colab (GPU CUDA) quanto localmente (Apple Silicon via MPS, ou CPU). Cada célula de código é marcada explicitamente como `[código de terceiros]`, `[adaptado de terceiros]` ou `[código próprio]`.

**Colab original (executado):** https://colab.research.google.com/drive/16U8Wvbz3wJHJcnk2sMmoc6ngYLIL0ezQ?usp=drive_link

## O artigo

[`artigo/artigo.pdf`](artigo/artigo.pdf) — formato IEEE dupla coluna, 6 páginas ([fonte em LaTeX](artigo/artigo.tex), compilável com [Tectonic](https://tectonic-typesetting.github.io/) via `tectonic artigo.tex`). Cobre descrição do problema, construção do corpus, arquitetura, resultados comparados, a busca automatizada de melhorias e uma proposta de extensão do projeto para responder perguntas no universo machadiano (recuperação aumentada por geração, RAG).

## Reproduzindo

O caminho mais simples é abrir o [notebook no Google Colab](https://colab.research.google.com/drive/16U8Wvbz3wJHJcnk2sMmoc6ngYLIL0ezQ?usp=drive_link) (ou [`notebook/projeto1_machado_gpt_Ronen_R_S_Filho_colab.ipynb`](notebook/projeto1_machado_gpt_Ronen_R_S_Filho_colab.ipynb) deste repositório) e executar todas as células, do início ao fim — o corpus é baixado e montado automaticamente, sem depender de mais nada.

Para rodar a busca de hiperparâmetros (`hpsearch/`) localmente, é necessário primeiro ter gerado `data/machado_char/input.txt` (feito pelo próprio notebook) dentro de um clone do nanoGPT; ver os scripts em `hpsearch/` e os comentários de cada um.

## Fontes e licença

O corpus é composto por obras de Machado de Assis em domínio público, obtidas do [Project Gutenberg](https://www.gutenberg.org/ebooks/author/9685) e do [Wikisource em português](https://pt.wikisource.org). O código deste repositório (exceto o nanoGPT, código de terceiros não incluído aqui — ver a seção acima) pode ser usado livremente para fins acadêmicos.
