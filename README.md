# Neural Architecture Comparison on Fashion-MNIST

A comparative study of four neural architectures — **ANN (MLP)**, **CNN**,
**U-Net**, and **Vision Transformer** — on the Fashion-MNIST image
classification benchmark, with training/validation curves, confusion matrices,
and classification reports for model evaluation.

MSc AI coursework — De Montfort University.

## Highlights

- **Four architectures on one benchmark**: ANN (256→128 hidden layers,
  dropout 0.3), CNN (32/64-channel 3×3 convolutions), U-Net
  (encoder–bottleneck–decoder), and Vision Transformer (patch size 7,
  embedding dim 64, 4 heads, 4 layers).
- **60,000 training images** (50k train / 10k validation) and **10,000 test
  images** across 10 fashion categories; normalization (0.5, 0.5), batch size
  128, 12 epochs per model on a T4 GPU.
- **CNN wins outright** with **90.72% test accuracy**, ~3.6pp ahead of the ANN
  baseline — consistent with convolutional inductive bias paying off on
  28×28 grayscale images.
- A from-scratch **Vision Transformer reaches 86.86%**, competitive despite
  no pre-training and a small dataset, but underperforms both the CNN and
  the simple ANN — the expected ViT data-hunger penalty.
- Full evaluation artefacts: per-epoch accuracy/loss curves, confusion
  matrices, and sklearn classification reports for every model.

## Results

All numbers below are the executed outputs of the evaluation cells in
`P2952028_Code_(SHEHRYAR_SHEHRYAR).ipynb` (test set, n = 10,000).

| Model | Test accuracy | Macro F1 | Best val accuracy |
| --- | --- | --- | --- |
| ANN (MLP 256/128) | 0.8713 | 0.8701 | 0.8881 |
| **CNN (2 conv blocks)** | **0.9072** | **0.9072** | 0.9180 |
| Vision Transformer | 0.8686 | 0.8669 | 0.8737 |

Key insight: the CNN dominates on both accuracy and macro-F1, with balanced
performance across classes (macro-F1 ≈ accuracy, so no class collapse). The
ANN plateaus at ~87–89% — a solid baseline. The ViT converges more slowly
(epoch 1: 57.7% train acc vs. 78.6% for the CNN) and never catches the CNN
within 12 epochs, illustrating why transformers typically need pre-training
or longer schedules on small datasets. U-Net was implemented
(encoder–bottleneck–decoder with [64, 128, 256, 512] feature channels) with
prediction visualizations; no classification metric was produced for it, so
it is excluded from the comparison table above.

## Methodology

1. **Dataset** — Fashion-MNIST via `torchvision` (auto-download), split
   50,000 train / 10,000 validation / 10,000 test, seeded (`seed = 42`) with
   `ToTensor` + `Normalize((0.5,), (0.5,))`.
2. **Training** — 12 epochs per model, batch size 128, on a T4 GPU;
   per-epoch train/val loss and accuracy logged and plotted.
3. **Architectures** — ANN: `Flatten → Linear(784→256) → Linear(256→128) →
   Linear(128→10)` with dropout 0.3; CNN: two 3×3 conv layers (32, 64
   channels) with pooling and dense head; U-Net: downsampling,
   bottleneck, and upsampling stages; ViT: 7×7 patch embedding (dim 64),
   4-head / 4-layer transformer encoder with CLS token.
4. **Evaluation** — test accuracy, sklearn `classification_report`
   (precision / recall / F1, macro and weighted averages), confusion
   matrices, and accuracy/loss curves per model.

## Project structure

```
├── P2952028_Code_(SHEHRYAR_SHEHRYAR).ipynb  # complete study: train, evaluate, visualize
├── P2952028-Summary (SHEHRYAR SHEHRYAR).pdf  # accompanying coursework summary
├── 2. Output/                                # figures / output artefacts
└── README.md
```

## Getting started

The notebook is self-contained and downloads Fashion-MNIST on first run.

```bash
pip install torch torchvision matplotlib scikit-learn
jupyter notebook P2952028_Code_(SHEHRYAR_SHEHRYAR).ipynb
```

Run cells top to bottom: environment setup → data loading → train ANN, CNN,
U-Net, ViT in turn → evaluate and compare. A GPU is recommended but not
required; all models train on 28×28 images in reasonable CPU time.

## Limitations & future work

- Single benchmark (Fashion-MNIST): conclusions about architecture choice do
  not automatically transfer to larger or color-image datasets.
- ViT trained from scratch for only 12 epochs — longer training,
  augmentation, or pre-training would likely narrow the CNN gap.
- Future work: hyperparameter sweeps, per-class error analysis, and
  transfer learning with pretrained vision models.

## Author

SHEHRYAR (P2952028) — MSc AI, De Montfort University.

## License

Coursework submitted to De Montfort University — shared for reference.
