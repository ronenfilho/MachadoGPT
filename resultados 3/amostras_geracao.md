# Amostras de geração de texto — Machado-GPT (15000 iterações)

Amostras geradas pelo modelo treinado nesta rodada (checkpoint da iteração 15000, corpus Gutenberg+Wikisource, ~13,9M caracteres — ver [README.md](README.md)). Geradas com `sample.py` do nanoGPT, parâmetros padrão do script (`temperature=0.8`, `top_k=200`), a partir do prompt **"Capitulo"**.

**Importante:** o texto abaixo é saída do modelo — geração autoregressiva caractere a caractere —, não um trecho original de nenhuma obra de Machado de Assis. Qualquer semelhança de vocabulário/estilo é o comportamento esperado (e desejado) de um modelo treinado sobre o corpus do autor.

---

## Amostra 1

> Capitulo III
>
> Era falso da paixão acceita. Era a outra que Rita casára e riu dalli a pouco; lá iriam commigo. Antes de me dizer não falar deste capitulo, voltaria depois, e provavelmente falaria della com muita ternura e communicação. Capitú, que o dissesse:
>
> --Quanto a mim, estou muito bonita.
>
> Era para i

## Amostra 2

> Capitulo VIII (Machado de Assis, Wikisource) =====
>
> A visita do cazamento era principalmente a minha mãe. Contei-lhe tudo o que senti ainda na minha parte. Não me lembro se fora de semelhante intervenção. Com effeito, á semelhança de acompanhar Emilia, fui ao theatro, dando ao corredor á esposa, á noite, á

## Amostra 3

> Capitulo. Os primeiros capitulos proprios disseram-me que eu lhe desse a mesma mocidade, e que vingança entre a minha e da fidelidade. Não vinha a casa de mana Rita, que estava passando na mesma semana. A familia era de mim uma daquellas ave-marias, um dos meus annos que lhe custavam a acceital-a; eu creio

---

## Leitura qualitativa

- **Melhor coerência local que `resultados 2/`**: a Amostra 1 produz "**Capitú**" — grafia correta (com acento) da personagem de *Dom Casmurro*, um dos nomes mais reconhecíveis de Machado — além de um diálogo bem formado (travessão + fala curta, pontuação correta).
- **Numeração de capítulo mais estável** (`Capitulo III`, `Capitulo VIII`) que na rodada anterior.
- **Artefato curioso na Amostra 2**: o modelo gerou `(Machado de Assis, Wikisource) =====` — um vazamento direto de metadado de página do Wikisource que sobreviveu à limpeza do corpus (provavelmente um cabeçalho de página mal filtrado em `build_corpus.py`). Vale investigar/reforçar a limpeza do corpus se aparecer com frequência — não invalida o resultado, mas é um ponto de atenção documentado aqui para transparência.
- **Mistura de ortografia de época e moderna** ("cazamento", "effeito", "theatro", "daquellas", "annos") convivendo com grafia atual — mesmo padrão das rodadas anteriores, esperado dado o corpus.
- **Coerência gramatical local boa, sem coerência narrativa de longo alcance** — mesma limitação estrutural das rodadas anteriores (esperada para um GPT em nível de caractere com `block_size=256`), não resolvida por mais iterações de treino.
