# Seq2Seq Chatbot — Architecture Comparison on Cornell Movie Dialogs

A comparative study of five sequence-to-sequence architectures — **MLP, RNN, GRU, LSTM,
and Transformer** — trained as a conversational chatbot on the **Cornell Movie-Dialogs Corpus**,
evaluated with **sacreBLEU** and token-level accuracy.

MSc AI coursework — De Montfort University.

## Highlights

- **Five architectures, one task**: MLP, vanilla RNN, GRU, LSTM, and Transformer seq2seq
  models (PyTorch; 256-dim embeddings, 512 hidden units, 2 layers) trained for 18 epochs
- **Real dialogue data**: 221,282 raw movie-dialogue pairs → 148,621 filtered
  (≤ 20 tokens per side) → 60,000 subsampled; 20,004-token vocabulary;
  48k / 6k / 6k train / validation / test split
- **Honest evaluation**: sacreBLEU plus token accuracy on held-out data, with a full
  validation comparison across all five models — **LSTM selected as best**
  (val accuracy 0.284), and the comparison shows how hard open-domain dialogue is
  (best BLEU 0.0083), an instructive result in itself
- **Interactive chat**: the best model runs as a command-line chatbot,
  with the trained weights (`best_model_LSTM.pt`) and vocabulary (`vocab.json`) saved
  for reuse

## Repository contents

| File | What it is |
|---|---|
| `P2952028_Code_(SHEHRYAR_SHEHRYAR).ipynb` | Full notebook: data download → cleaning → vocab → 5 models → BLEU evaluation → chat |
| `P2952028-Summary (SHEHRYAR SHEHRYAR).pdf` | Coursework summary report |
| `2. Output/` | Training curves, BLEU/accuracy comparisons, and chat screenshots |

## Run it

```bash
pip install torch nltk sacrebleu tqdm matplotlib
```

Open the notebook in Colab (GPU recommended) and run all cells. The Cornell Movie-Dialogs
Corpus downloads automatically. After training, the last cells start an interactive chat
session with the best model.

## Tech stack

`Python` · `PyTorch` · `NLTK` · `sacreBLEU` · `pandas` · `matplotlib`

## Author

**SHEHRYAR** — MSc Artificial Intelligence, De Montfort University, Leicester.
