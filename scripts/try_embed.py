from app.embeddings import cosine_similarity, embed

sentences = [
    "I built machine learning pipelines in production",   # A
    "Experience with MLOps and deploying ML models",      # B: milta-julta matlab, alag words
    "I enjoy cooking Italian pasta",                      # C: bilkul alag topic
    "Docker",                                             # D
    "containerization",                                   # E: D jaisa matlab
]

vectors = embed(sentences)
print("Vector shape:", vectors.shape, "(sentences, numbers per sentence)")
print("First 5 numbers of A:", vectors[0][:5].round(3))

print()
print(f"A vs B (same meaning):   {cosine_similarity(vectors[0], vectors[1]):.2f}")
print(f"A vs C (different):      {cosine_similarity(vectors[0], vectors[2]):.2f}")
print(f"Docker vs containerization: {cosine_similarity(vectors[3], vectors[4]):.2f}")
print(f"Docker vs pasta:            {cosine_similarity(vectors[3], vectors[2]):.2f}")