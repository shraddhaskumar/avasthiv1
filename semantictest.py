from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load semantic search model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Example queries
queries = ["How to handle stress?", "Methods to manage anxiety", "Best smartphones", "History of AI"]

# Generate embeddings
embeddings = model.encode(queries)

# Reduce dimensionality for visualization
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# Plot the results
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])

for i, query in enumerate(queries):
    plt.annotate(query, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))

plt.title("Semantic Search Embedding Clusters")
plt.show()
