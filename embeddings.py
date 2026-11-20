"""
Embedding Generation Module
===========================
Initializes the Hugging Face embedding model (sentence-transformers/all-MiniLM-L6-v2)
to transform text chunks into numerical dense vectors.
"""

import os
import socket
from langchain_huggingface import HuggingFaceEmbeddings


def _check_network_and_configure_cache(host: str = "huggingface.co", port: int = 443, timeout: float = 0.5) -> None:
    """
    Check if Hugging Face Hub is reachable. If not or if working offline,
    enable HF_HUB_OFFLINE so cached local models load immediately without connection retries.
    """
    if os.getenv("HF_HUB_OFFLINE") == "1":
        return

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host, port))
        sock.close()
    except (socket.timeout, OSError):
        # Offline or sandbox without network: use cached local snapshot
        os.environ["HF_HUB_OFFLINE"] = "1"


def get_embedding_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> HuggingFaceEmbeddings:
    """
    Initialize and return a Hugging Face embedding model.

    Args:
        model_name (str): Identifier on the Hugging Face Hub.
                          Defaults to 'sentence-transformers/all-MiniLM-L6-v2'.

    Returns:
        HuggingFaceEmbeddings: Initialized LangChain embeddings wrapper.
    """
    _check_network_and_configure_cache()

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


if __name__ == "__main__":
    print("Testing embedding model initialization...")
    embeddings = get_embedding_model()
    sample_vector = embeddings.embed_query("19th century Indian railways")
    print(f"Model: sentence-transformers/all-MiniLM-L6-v2")
    print(f"Embedding vector dimension: {len(sample_vector)}")
