from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import normalize


def average_embeddings(documents,
                       batch_size=32,
                       model_max_length=512,
                       embedding_model='sentence-transformers/all-MiniLM-L6-v2'):
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    model = AutoModel.from_pretrained(embedding_model)

    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    device = torch.device(device)

    data_loader = DataLoader(documents, batch_size=batch_size, shuffle=False)

    model.eval()
    model.to(device)

    average_embeddings = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Embedding vocabulary"):
            # Tokenize the batch with padding and truncation
            batch_inputs = tokenizer(
                batch,
                padding="max_length",
                max_length=model_max_length,
                truncation=True,
                return_tensors="pt"
            )

            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            last_hidden_state = model(**batch_inputs).last_hidden_state
            avg_embedding = last_hidden_state.mean(dim=1)
            average_embeddings.append(avg_embedding.cpu().numpy())

    document_vectors = normalize(np.vstack(average_embeddings))

    return document_vectors


def contextual_token_embeddings(documents,
                                batch_size=32,
                                model_max_length=512,
                                embedding_model='sentence-transformers/all-MiniLM-L6-v2'):
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    model = AutoModel.from_pretrained(embedding_model)

    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    device = torch.device(device)

    # DataLoader to process the documents in batches
    data_loader = DataLoader(documents, batch_size=batch_size, shuffle=False)

    model.eval()
    model.to(device)

    last_hidden_states = []
    all_attention_masks = []
    all_tokens = []

    # Embed documents batch-wise
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Embedding documents"):
            # Tokenize the batch with padding and truncation
            batch_inputs = tokenizer(
                batch,
                padding="max_length",
                max_length=model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            all_attention_masks.extend(batch_inputs['attention_mask'])
            all_tokens.extend(batch_inputs['input_ids'])
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            last_hidden_state = model(**batch_inputs).last_hidden_state
            last_hidden_states.append(last_hidden_state.cpu())

    # Concatenate the embeddings from all batches
    all_hidden_states = torch.cat(last_hidden_states, dim=0)

    document_token_embeddings = []
    document_tokens = []
    document_labels = []

    for ind, (hidden_state, attention_mask, tokens) in enumerate(
            zip(all_hidden_states, all_attention_masks, all_tokens)):
        embeddings = hidden_state[attention_mask.nonzero(as_tuple=True)]
        tokens = tokens[attention_mask.nonzero(as_tuple=True)]
        tokens = [tokenizer.decode(token) for token in tokens]

        document_token_embeddings.append(embeddings.detach().numpy())
        document_tokens.append(tokens)
        document_labels.extend([ind] * len(tokens))

    return document_token_embeddings, document_tokens, document_labels


def sliding_window_average(document_token_embeddings, document_tokens, window_size, stride):
    # Store the averaged embeddings
    averaged_embeddings = []
    chunk_tokens = []

    # Iterate over each document
    for doc, tokens in tqdm(zip(document_token_embeddings, document_tokens)):
        doc_averages = []

        # Slide the window over the document with the specified stride
        for i in range(0, len(doc), stride):

            start = i
            end = i + window_size

            if start != 0 and end > len(doc):
                start = len(doc) - window_size
                end = len(doc)

            window = doc[start:end]

            # Calculate the average embedding for the current window
            window_average = np.mean(window, axis=0)

            doc_averages.append(window_average)
            chunk_tokens.append(" ".join(tokens[start:end]))

        averaged_embeddings.append(doc_averages)

    averaged_embeddings = np.vstack(averaged_embeddings)
    averaged_embeddings = normalize(averaged_embeddings)

    return averaged_embeddings, chunk_tokens


def average_adjacent_tokens(token_embeddings, window_size):
    num_tokens, embedding_size = token_embeddings.shape
    averaged_embeddings = np.zeros_like(token_embeddings)

    token_embeddings = normalize(token_embeddings)

    # Define the range to consider based on window_size
    for i in range(num_tokens):
        start_idx = max(0, i - window_size)
        end_idx = min(num_tokens, i + window_size + 1)

        # Compute the average for the current token within the specified window
        averaged_embeddings[i] = np.mean(token_embeddings[start_idx:end_idx], axis=0)

    return averaged_embeddings


def smooth_document_token_embeddings(document_token_embeddings, window_size=2):
    smoothed_document_embeddings = []

    for doc in tqdm(document_token_embeddings, desc="Smoothing document token embeddings"):
        smoothed_doc = average_adjacent_tokens(doc, window_size=window_size)
        smoothed_document_embeddings.append(smoothed_doc)

    return smoothed_document_embeddings
