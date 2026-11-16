from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

sentences = [
    "The borrower may exit the loan during the cooling-off period without penalty.",
    "A customer can close the loan in the first few days and pay no penalty.",
    "The Board must approve a policy for lending against gold collateral.",
]

vectors = model.encode(sentences, normalize_embeddings=True)

print("Shape (sentences, numbers each):", vectors.shape)
print("Sentence 1 vs 2:", float(vectors[0] @ vectors[1]))
print("Sentence 1 vs 3:", float(vectors[0] @ vectors[2]))
print("Longest text the model reads:", model.max_seq_length, "tokens")