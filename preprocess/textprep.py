import re
import torch
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Text Preprocessing
def clean_text(text: str) -> str:
    # Remove all symbols except full stops and commas
    cleaned_text = re.sub(r'[^\w\s.,]', '', text)
    # Remove extra whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

# Text Summarization
def summarize_text(text: str, model_name='all-MiniLM-L6-v2', ratio=0.4) -> str:
    # Split the text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Generate embeddings for each sentence
    model = SentenceTransformer(model_name)
    sentence_embeddings = model.encode(sentences)
    
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(sentence_embeddings)
    
    # Calculate sentence scores
    scores = similarity_matrix.sum(axis=1)
    
    # Select top sentences
    top_sentences_indices = scores.argsort()[-int(len(sentences) * ratio):]
    top_sentences_indices = sorted(top_sentences_indices)
    
    # Combine selected sentences
    summarized_text = ' '.join([sentences[i] for i in top_sentences_indices])
    return summarized_text

# Embedding Generation
def generate_embeddings(text: str, model_name='bert-base-uncased'):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the embeddings from the [CLS] token
    embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings

# Example usage
text = "Your article text here. It should be a longer piece of text with multiple sentences to demonstrate the summarization effectively."
cleaned_text = clean_text(text)
summarized_text = summarize_text(cleaned_text)
embeddings = generate_embeddings(summarized_text)

print("Cleaned Text:", cleaned_text)
print("Summarized Text:", summarized_text)
print("Embeddings shape:", embeddings.shape)