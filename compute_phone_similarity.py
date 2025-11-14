import torch
from utils.fft_dataset import get_one_of_each

ds = get_one_of_each()
print(ds)

# Calculate pairwise cosine similarity between phone embeddings
embeddings = []
for i in range(len(ds)):
    data, label = ds[i]
    embeddings.append(data.flatten())
embeddings = torch.stack(embeddings)  # Shape: (num_phones, embedding_dim)
embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)  # Normalize
similarity_matrix = torch.matmul(embeddings, embeddings.T)  # Shape: (num_phones, num_phones)
print("Pairwise Cosine Similarity Matrix:")
print(similarity_matrix)