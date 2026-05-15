from langchain_huggingface import HuggingFaceEmbeddings

from core.config import settings
from core.utils.device import resolve_device


def get_embedding_function():
    device = resolve_device(settings.EMBEDDING_DEVICE)
    # 128-batch is tuned for a 48GB GPU; on CPU it just wastes memory without
    # speeding anything up. 32 is a reasonable laptop-CPU default.
    batch_size = 128 if device == "cuda" else 32
    return HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1.5",
        model_kwargs={
            "device": device,
            "trust_remote_code": True,
        },
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": batch_size,
        },
        # nomic-embed-text-v1.5 requires task-specific prefixes for optimal embeddings.
        # "prompt" is passed to sentence_transformers.encode() and prepended to text.
        # query_encode_kwargs applies ONLY to embed_query() calls (search-time),
        # NOT to embed_documents() calls (index-time — handled in vectorstore.py).
        query_encode_kwargs={
            "prompt": "search_query: ",
        },
    )
