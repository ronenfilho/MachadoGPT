#!/bin/bash
# Roda as 3 avaliações completas de hpsearch/ em sequência (não em paralelo — evita disputar o
# mesmo device MPS e distorcer os tempos/MFU medidos). Uso: nohup ./run_full_evaluations.sh &
# (ou via run_in_background do Bash tool). Log completo vai para full_run.log neste diretório.
set -e
cd "$(dirname "$0")"
source "../../Tutorial/01 - Introduction to PyTorch/.venv/bin/activate"

echo "=== INÍCIO $(date) ==="

echo ""
echo "########## 1/3: run_search.py (arquitetura + hiperparâmetros) ##########"
python3 run_search.py

echo ""
echo "########## 2/3: run_embedding_comparison.py (posição) ##########"
python3 run_embedding_comparison.py

echo ""
echo "########## 3/3: run_tokenization_comparison.py (char vs BPE) ##########"
python3 run_tokenization_comparison.py

echo ""
echo "=== FIM $(date) ==="
