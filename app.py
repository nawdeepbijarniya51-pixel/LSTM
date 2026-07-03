"""
LSTM Next-Word / Sequence Generator — FastAPI Backend
-------------------------------------------------------
Endpoints:
    GET  /health           -> simple health check
    POST /predict-next     -> one sampled next word for given text (matches
                              the notebook's pred(): weighted random choice
                              over the full temperature-scaled distribution,
                              not a ranked top-K)
    POST /generate         -> generate N words in one shot (JSON response)
    GET  /generate-stream  -> generate N words, streamed via Server-Sent Events
                              (used for the live progress bar / typewriter UI)

Model details (auto-detected from model_lstm.h5):
    Embedding(12000, 50) -> LSTM(128) -> Dense(12000, softmax)
    Input shape: (None, 65)   -> maxlen = 65, padding = 'pre'
    Output shape: (None, 12000)
"""

import pickle
import json
import asyncio
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Load model + tokenizer + maxlen once at startup
# ---------------------------------------------------------------------------
print("Loading tokenizer...")
with open(BASE_DIR / "tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

print("Loading maxlen...")
with open(BASE_DIR / "maxlen.pkl", "rb") as f:
    MAXLEN = pickle.load(f)  # 65

print("Loading LSTM model...")
model = keras.models.load_model(BASE_DIR / "model_lstm.h5")
print("Model loaded. Input shape:", model.input_shape, "Output shape:", model.output_shape)

VOCAB_SIZE = model.output_shape[-1]

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="LSTM Next-Word Predictor API")

# Allow your GitHub Pages frontend (and local dev) to call this API.
# Tighten allow_origins to your real frontend URL before going to production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Core inference helpers
# ---------------------------------------------------------------------------
def _encode(text: str) -> np.ndarray:
    """Tokenize + pad a raw text string to feed into the model.
    NOTE: the notebook lowercases the seed text before tokenizing
    (`text = text.lower()` inside `pred()`), so we must do the same here
    or words that don't match the tokenizer's vocab casing become <OOV>
    and quietly wreck predictions.
    """
    text = text.lower()
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAXLEN, padding="pre")
    return padded


def _unknown_words(text: str) -> list:
    """Diagnostic helper: Keras' Tokenizer silently DROPS any word that
    isn't in its vocab (unless it was fit with oov_token=...) — it does not
    substitute a placeholder. That means a seed sentence full of unknown
    words can encode down to an (almost) all-zero/padding sequence, so the
    model effectively ignores your input and returns near-identical, near-
    uniform predictions every time. This lists which words got dropped so
    that's visible instead of silent.
    """
    return [w for w in text.lower().split() if w not in tokenizer.word_index]


def _softmax_with_temperature(preds: np.ndarray, temperature: float) -> np.ndarray:
    """Re-scale a softmax distribution by temperature, matching the notebook's
    `pred()` function exactly:
        preds = np.log(preds + 1e-8) / temperature
        preds = np.exp(preds) / np.sum(np.exp(preds))
    temperature == 1   : model's original distribution
    temperature -> 0   : sharper / more deterministic
    temperature -> 2   : flatter / more random
    """
    temperature = max(temperature, 1e-3)
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    return exp_preds / np.sum(exp_preds)


def _top_k(preds: np.ndarray, k: int = 5):
    top_idx = np.argsort(preds)[-k:][::-1]
    return [
        {"word": tokenizer.index_word.get(int(i), "<OOV>"), "probability": float(preds[i])}
        for i in top_idx
    ]


def predict_next_word(text: str, temperature: float = 0.8):
    """Mirrors the notebook's pred() exactly: one word, picked by weighted
    random sampling over the full temperature-scaled distribution — not a
    ranked top-K. Also returns that word's own probability for context.
    """
    padded = _encode(text)
    raw_preds = model.predict(padded, verbose=0)[0]
    scaled = _softmax_with_temperature(raw_preds, temperature)
    probs = scaled / scaled.sum()
    next_idx = int(np.random.choice(len(probs), p=probs))
    next_word = tokenizer.index_word.get(next_idx, "<OOV>")
    return next_word, float(probs[next_idx])


def generate_sequence(seed_text: str, num_words: int, temperature: float = 0.8, top_k: int = 5):
    """Generator that yields one dict per generated word:
    {word, probability, step, total, top_k}

    Mirrors the notebook's pred()/pred_sq() loop exactly: always sample
    from the temperature-scaled distribution via np.random.choice — no
    greedy argmax fallback, since the notebook never does that.
    """
    current_text = seed_text.strip()
    for step in range(1, num_words + 1):
        padded = _encode(current_text)
        raw_preds = model.predict(padded, verbose=0)[0]
        scaled = _softmax_with_temperature(raw_preds, temperature)

        # normalize defensively against floating point drift, then sample
        # exactly like the notebook's np.random.choice(len(preds), p=preds)
        probs = scaled / scaled.sum()
        next_idx = int(np.random.choice(len(probs), p=probs))

        next_word = tokenizer.index_word.get(next_idx, "<OOV>")
        candidates = _top_k(probs, k=top_k)

        current_text = f"{current_text} {next_word}".strip()

        yield {
            "word": next_word,
            "step": step,
            "total": num_words,
            "progress": round(step / num_words * 100, 2),
            "generated_text": current_text,
            "top_k": candidates,
        }


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Seed text")
    temperature: float = Field(0.8, ge=0.0, le=2.0)


class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=1)
    num_words: int = Field(20, ge=1, le=200)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: int = Field(5, ge=1, le=15)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok", "vocab_size": int(VOCAB_SIZE), "maxlen": int(MAXLEN)}


@app.post("/api/predict-next")
def predict_next(req: PredictRequest):
    word, probability = predict_next_word(req.text, temperature=req.temperature)
    return {
        "seed_text": req.text,
        "word": word,
        "probability": probability,
        "unknown_words": _unknown_words(req.text),
    }


@app.post("/api/generate")
def generate(req: GenerateRequest):
    """Non-streaming version: returns the full generated text + per-step trace."""
    steps = list(
        generate_sequence(req.text, req.num_words, temperature=req.temperature, top_k=req.top_k)
    )
    final_text = steps[-1]["generated_text"] if steps else req.text
    return {"seed_text": req.text, "generated_text": final_text, "steps": steps}


@app.get("/api/generate-stream")
async def generate_stream(
    text: str = Query(..., min_length=1),
    num_words: int = Query(20, ge=1, le=200),
    temperature: float = Query(0.8, ge=0.0, le=2.0),
    top_k: int = Query(5, ge=1, le=15),
):
    """Server-Sent Events stream: one 'data:' event per generated word.
    The frontend uses this to drive a real (not fake) progress bar.
    """

    async def event_generator():
        for item in generate_sequence(text, num_words, temperature=temperature, top_k=top_k):
            yield f"data: {json.dumps(item)}\n\n"
            # tiny yield so the event loop can flush each chunk to the client
            await asyncio.sleep(0)
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)