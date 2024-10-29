import streamlit as st
import torch
import json
import numpy as np
from torch import nn
import nltk

# Download nltk tokenizer
nltk.download('punkt')

# Define the RNNTextGenerator class
class RNNTextGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNNTextGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])
        return x

# Helper function to load model and vocab
def load_model_and_vocab(model_path, vocab_path):
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    model = RNNTextGenerator(len(vocab_data["word_to_idx"]), st.session_state.embedding_dim, st.session_state.hidden_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model, vocab_data["word_to_idx"], vocab_data["idx_to_word"]

# Function to preprocess user input text
def preprocess_text(text, word_to_idx):
    tokens = nltk.tokenize.word_tokenize(text.lower())
    words = [word for word in tokens if word.isalpha()]
    indices = [word_to_idx.get(word, word_to_idx['unknown token']) for word in words]
    return indices

# Streamlit app
st.title("Text Generation App")
st.write("Generate the next word predictions based on a pre-trained RNN model.")

# Dropdowns to select the model configuration
context_length = st.selectbox("Select context length", [3, 4])
k_value = st.selectbox("Select k-value", [1, 3])
random_seed = st.selectbox("Select random seed", [42, 123])

# Select model path based on configuration
model_path = f'rnn_text_generator_c{context_length}_k{k_value}_seed{random_seed}.pth'
vocab_path = f'vocab_c{context_length}_k{k_value}_seed{random_seed}.json'

# Model parameters
embedding_dim = st.slider("Embedding Dimension", 32, 128, 64)
hidden_dim = st.slider("Hidden Dimension", 64, 256, 128)

# Load the model and vocabulary
model, word_to_idx, idx_to_word = load_model_and_vocab(model_path, vocab_path)

# Input text from the user
user_text = st.text_input("Enter the starting text:")
k_predictions = st.number_input("Number of words to predict", min_value=1, max_value=10, value=1)

if st.button("Generate"):
    if user_text:
        input_indices = preprocess_text(user_text, word_to_idx)
        
        # Ensure the context length is satisfied by padding with "padding" tokens
        if len(input_indices) < context_length:
            input_indices = [word_to_idx["padding"]] * (context_length - len(input_indices)) + input_indices
        else:
            input_indices = input_indices[-context_length:]

        # Convert input to tensor and generate predictions
        input_tensor = torch.tensor([input_indices], dtype=torch.long)
        output_text = user_text

        for _ in range(k_predictions):
            with torch.no_grad():
                output = model(input_tensor)
                _, next_word_idx = torch.max(output, dim=1)
                next_word = idx_to_word[next_word_idx.item()]

                # Append to output text
                output_text += ' ' + next_word

                # Update input tensor
                input_indices = input_indices[1:] + [next_word_idx.item()]
                input_tensor = torch.tensor([input_indices], dtype=torch.long)

        # Display the generated text
        st.write("Generated Text:")
        st.write(output_text)
    else:
        st.write("Please enter a starting text.")
