# Amostras de geração de texto — Machado-GPT (corpus maximizado)

Amostras geradas pelo modelo treinado nesta rodada (checkpoint da iteração 5000, corpus Gutenberg+Wikisource, ~13,9M caracteres — ver [README.md](README.md)). Geradas com `sample.py` do nanoGPT, parâmetros padrão do script (`temperature=0.8`, `top_k=200`), a partir do prompt **"Capitulo"**.

**Importante:** o texto abaixo é saída do modelo — geração autoregressiva caractere a caractere —, não um trecho original de nenhuma obra de Machado de Assis. Qualquer semelhança de vocabulário/estilo é o comportamento esperado (e desejado) de um modelo treinado sobre o corpus do autor.

---

## Amostra 1

> Capitulo respondeu que um dos seus olhos me casassem de contever com a eengraça de ver em alguma coisa de imprecação. Mas o que eu lhe é que sentira a vontade da sua parte e não sacrificá-la pelas duas circunstâncias de Helena; a este senhor passa-me por si que não vejo alguma coisa por alguns segundos a minha mãe. Pois o que é que eu possa dizer nada, mas se tenho voltado ao contrário.
>
> Mas ainda um meio

## Amostra 2

> Capitulo sentia-se de se mettido em sair da cadeira de Luiz Garcia. Era a mão do seu filho. Mas, quem ella me dizia estes dous annos? Acabava de fazer de casa; mas, uma vez que o descobria em outro lugar com a viuva, debalde de fazer o impulso, e pediu um pequeno para todo o caso.
>
> CAPITULO CLXXVIII
>
> --Tabuleta
>
> Meu pae o amor escreveu a carta de Helena e galante com que não houvessem ficar em casa. V

## Amostra 3

> Capitulo sahiu do jornal e descia alli sahindo para a mesa, e a segunda lhe custou ser a conversação da cabeça. Jorge sentiu que elle a moça a precisava de sua mãe. Donde pedira essa situação mais atual affectuosa. Nem por isso mesmo, nem a constipação das carnes pareciam apertar muito ás duas cousas alterações de exigencia e de namorados com que o reproduziu no pretendimento da opposição.
>
> Rodrigo cedeu

---

## Leitura qualitativa

- **Vocabulário mais rico que a rodada anterior** (`resultados 1/`): aparecem nomes de personagens reais de Machado ("Helena", "Luiz Garcia") e uma numeração de capítulo em algarismos romanos no estilo real das obras (`CAPITULO CLXXVIII`) — reflexo direto do corpus ~3,5x maior e muito mais diverso (crônicas, teatro, correspondência, além dos romances).
- **Mistura de ortografia de época e moderna** ("mettido", "annos", "affectuosa", "cousas" convivendo com grafias mais modernas) — esperado, já que o Wikisource inclui tanto transcrições históricas quanto algumas páginas com ortografia atualizada.
- Mantém o maneirismo de diálogo de Machado (travessão + fala curta, ex. `--Tabuleta`).
- **Coerência gramatical local boa, sem coerência narrativa de longo alcance** — mesma limitação estrutural já observada em `resultados 1/` (esperada para um GPT em nível de caractere com `block_size=256`), não resolvida só por aumentar o corpus.
