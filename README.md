# Transformer
# Transformer from Scratch ( Italian to English Translation)

A PyTorch implementation of the sequence-to-sequence Transformer architecture built from scratch, based on the landmark paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762). 

This project trains a custom Transformer model on the **Opus Books** dataset to perform machine translation ( Italian → English).

---

##  Project Architecture & Features

- **Built from scratch:** Implements Multi-Head Attention, Positional Encoding, Encoder-Decoder blocks, Projection Layers, and Masking mechanisms without using `torch.nn.Transformer`.
- **Dynamic Tokenization:** Uses Hugging Face `tokenizers` to build language-specific subword tokenizers dynamically (`tokenizer_en.json`, `tokenizer_it.json`).
- **Configurable Pipeline:** Standardized training workflow managed via `vconfig.py`.
- **Experiment Tracking:** Integrated with TensorBoard for monitoring loss and metrics during training.

---

##  Repository Structure

```text
VTRANSFORMER/
├── opus_books_weights/  # Checkpoints and saved model weights
├── runs/                # TensorBoard logs
├── tokenizer_en.json    # Trained English subword tokenizer
├── tokenizer_it.json    # Trained Italian subword tokenizer
├── vconfig.py           # Configuration dictionary for model & training hyperparameters
├── vdataset.py          # PyTorch Dataset implementation for bilingual sequence pairs
├── vmodel.py           # Core Transformer architecture (Encoder, Decoder, Attention)
└── Vtrain.py            # Main training and evaluation loop
```

---

##  Model Architecture & Hyperparameters

The default configurations used in `vconfig.py` and `vmodel.py`:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| Model Dimension ($d_{\text{model}}$) | 512 | Dimensionality of input embeddings and hidden states |
| Layers ($N$) | 6 | Number of Encoder and Decoder blocks |
| Attention Heads ($h$) | 8 | Number of parallel attention heads |
| Feed-Forward Dim ($d_{ff}$) | 2048 | Dimension of the inner dense layer |
| Dropout | 0.1 | Dropout probability across sub-layers |
| Sequence Length | 350 | Maximum token sequence length |
| Batch Size | 8 | Training batch size |
| Learning Rate | 1e-4 | Learning rate for the Adam optimizer |
| Dataset | opus_books | Source corpus (en $\rightarrow$ it) |

---

##  Quick Start

### 1. Requirements
Ensure you have Python 3.8+ and PyTorch installed. Install additional dependencies:

```bash
pip install torch datasets tokenizers tensorboard
```

### 2. Training the Model
To train the Transformer model from scratch, execute:

```bash
python Vtrain.py
```

The script automatically checks for existing tokenizers/weights and creates them if they don't exist.

### 3. Monitoring Training
Visualize training loss and metrics via TensorBoard:

```bash
tensorboard --logdir=runs
```

---

##  Acknowledgments & References

- **Paper:** Vaswani et al., ["Attention Is All You Need" (2017)](https://arxiv.org/abs/1706.03762)
- Implemented for educational purposes to understand low-level PyTorch tensor operations and multi-head attention mechanics.
