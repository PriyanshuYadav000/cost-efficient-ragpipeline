from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL
model = None

def get_model():
    global model
    if model is None:
        model = SentenceTransformer(EMBEDDING_MODEL)
    return model


def embed_texts(texts):
    return get_model().encode(texts,normalize_embeddings=True).tolist()


def embed_query(query):
    return get_model().encode([query],normalize_embeddings=True)[0].tolist()
