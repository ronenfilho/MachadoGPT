# Amostras de geração de texto — Machado-GPT

Amostras geradas pelo modelo treinado (checkpoint da iteração 5000, `always_save_checkpoint=True` — ver [README.md](README.md) para a discussão sobre esse checkpoint vs. o de melhor val loss). Geradas com `sample.py` do nanoGPT, parâmetros padrão do script (`temperature=0.8`, `top_k=200`), a partir do prompt **"Capitulo"**.

**Importante:** o texto abaixo é saída do modelo — geração autoregressiva caractere a caractere —, não um trecho original de nenhuma obra de Machado de Assis. Qualquer semelhança de vocabulário/estilo é o comportamento esperado (e desejado) de um modelo treinado sobre o corpus do autor.

---

## Amostra 1

> Capitulo (deixou os dedos) e depois, o coronel era a mesma cousa, o portador delle, a affeição divina gasta, a variedade era tambem uma para o _nosso_ de sua maneira.» Era um incidente unico. Era muito bem bonito; a ideia de um regresso da morte attenção, porque elle foi boa vontade de o amigo imperial. Mas o bacharel acompanhou-o com a graça, na esperança do seu amigo de seus paes e o desespero, que el

## Amostra 2

> Capitulo estava pedido para mim, e para as minhas idéas a morte, da filha e da mãe, e a recordação antiga estive no regaço da familia. A dor da irmã viera do seu pae, tanto que fosse o que é o pensamento que a minha mãe suspira. Não sei o que diga que era a sua impossibilidade resistia em que pudesse não dizer alguma cousa a de desconfiança de que alli essa alma solitaria e intima. Eu a saber a questão

## Amostra 3

> Capitulo dessa causa accentuou o mesmo beneficio.
>
> --Você é boa, disse o conselheiro.
>
> --Deve ser minha paixão.
>
> --Pobre moça! exclamou elle.
>
> --Não tenho com o seu queixo.
>
> --Está o que era, almoçou por dona da casa. O que fizesse a Sandice é uma boa Sophia e um pretexto. Até aqui insistiu; perguntei tudo, por amor que a senhora retrospectiva é que lhe chame a conta da morte, e esta segunda gravidade d

---

## Leitura qualitativa

- **Ortografia e sintaxe de época:** o modelo reproduz consistentemente a grafia pré-reforma ortográfica usada por Machado ("elle", "delle", "cousa", "affeição", "attenção"), sem que isso tenha sido imposto por regra — é aprendido diretamente da distribuição de caracteres do corpus.
- **Maneirismo de diálogo:** a Amostra 3 captura bem o padrão característico de diálogo de Machado — travessão seguido de fala curta e verbo dicendi (`--Você é boa, disse o conselheiro.`) —, incluindo a alternância entre falas e comentário do narrador.
- **Coerência gramatical local:** boa. Concordância verbal/nominal e estrutura de frase em geral corretas dentro de janelas curtas (uma ou poucas orações).
- **Limitação: sem coerência narrativa de longo alcance.** Não há enredo persistente, personagens que se mantêm ao longo do trecho, ou continuidade lógica além de algumas frases. Isso é esperado para um GPT em nível de caractere deste porte (10,67M parâmetros) e contexto (`block_size=256` caracteres, ~40-50 palavras) — não é um defeito da implementação, é uma limitação estrutural do escopo do experimento (ver seção de limitações em [README.md](README.md)).
