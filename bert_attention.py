from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# Load tokenizer and model with attention outputs
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = TFBertModel.from_pretrained("bert-base-uncased", output_attentions=True)

# Sample sentence
text = "the movie was very emotional and well directed"
inputs = tokenizer(text, return_tensors="tf")
outputs = model(**inputs)

# Extract attention
attentions = outputs.attentions  # List of 12 layers

# Choose layer 0, head 0
attention = attentions[0][0, 0].numpy()
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(attention[:len(tokens), :len(tokens)], xticklabels=tokens, yticklabels=tokens, cmap="Blues")
plt.title("BERT Attention Map (Layer 0, Head 0)")
plt.xticks(rotation=90)
plt.show()
