"""Embedding model: multilingual-e5-base, running as quantized ONNX (no PyTorch).

Model + tokenizer files come from the Xenova/multilingual-e5-base repo, which
already ships ready-to-use ONNX exports. They're downloaded once via
huggingface_hub (cached locally after that) and run with onnxruntime.

E5 models expect a "query: " / "passage: " prefix depending on which side of
retrieval the text is on, and use mean pooling + L2 normalization over the
last hidden state (not the pooler output).
"""

from functools import lru_cache

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

REPO_ID = "Xenova/multilingual-e5-base"
ONNX_FILE = "onnx/model_quantized.onnx"
TOKENIZER_FILE = "tokenizer.json"
MAX_LENGTH = 512
PAD_TOKEN = "<pad>"
PAD_ID = 1


@lru_cache(maxsize=1)
def _session() -> ort.InferenceSession:
    path = hf_hub_download(REPO_ID, ONNX_FILE)
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


@lru_cache(maxsize=1)
def _tokenizer() -> Tokenizer:
    path = hf_hub_download(REPO_ID, TOKENIZER_FILE)
    tok = Tokenizer.from_file(path)
    tok.enable_padding(pad_id=PAD_ID, pad_token=PAD_TOKEN)
    tok.enable_truncation(max_length=MAX_LENGTH)
    return tok


def _embed(texts: list[str]) -> np.ndarray:
    tok = _tokenizer()
    session = _session()
    encodings = tok.encode_batch(texts)

    input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
    feeds = {"input_ids": input_ids, "attention_mask": attention_mask}

    input_names = {i.name for i in session.get_inputs()}
    if "token_type_ids" in input_names:
        feeds["token_type_ids"] = np.zeros_like(input_ids)

    [last_hidden_state] = session.run(None, feeds)

    mask = attention_mask[:, :, None].astype(np.float32)
    summed = (last_hidden_state * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    pooled = summed / counts

    norms = np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9, None)
    return pooled / norms


def embed_passages(texts: list[str]) -> np.ndarray:
    """Embed a batch of document chunks for storage. Returns (n, 768) float32."""
    return _embed([f"passage: {t}" for t in texts])


def embed_query(text: str) -> np.ndarray:
    """Embed a single search query. Returns a (768,) float32 vector."""
    return _embed([f"query: {text}"])[0]
