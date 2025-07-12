# BERT vs RNN + Attention on IMDB Text Classification 📊

This project compares:
- A custom RNN + Attention model (TensorFlow)
- BERT Transformer (Hugging Face)

We also visualize their attention maps to understand how the models focus on different words.

## What You'll Learn

- How attention works in RNN vs BERT
- How to visualize attention weights
- Accuracy comparison of both models on sentiment classification

## 📈 Results

| Model           | Accuracy | Notes |
|----------------|----------|-------|
| RNN + Attention| 85%      | Fast, custom attention |
| BERT Base      | 90–92%   | Slower but more accurate |

## 📊 Heatmap Samples

### RNN + Attention

<img width="1600" height="300" alt="rnn" src="https://github.com/user-attachments/assets/ff949287-d0b0-448d-a99a-423e606dbc36" />

### BERT Attention (Layer 0, Head 0)
<img width="1000" height="800" alt="bert" src="https://github.com/user-attachments/assets/737ac1d9-95e9-4103-bfd5-d1d7081e7e0f" />


## 🚀 Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Train RNN model
python rnn_attention/train_rnn_attention.py

# Visualize RNN attention
python rnn_attention/visualize_attention.py

# Visualize BERT attention
python bert_attention/visualize_bert_attention.py
