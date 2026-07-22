import json
import os
import re
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict

import pandas as pd
import mlflow.sklearn
import chromadb
from anthropic import Anthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPORT_DIR = PROJECT_ROOT / 'model_export'
CHROMA_DIR = PROJECT_ROOT / 'chroma_data'

SENSOR_ID_PATTERN = re.compile(r'sensor[\s_]?(\d{1,2})', re.IGNORECASE)

RAG_TOP_K = 4
CLAUDE_MODEL = 'claude-sonnet-5'

model_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not EXPORT_DIR.exists():
        raise RuntimeError(f'{EXPORT_DIR} not found — run export_model.py first')

    model_state['model'] = mlflow.sklearn.load_model(str(EXPORT_DIR / 'model'))
    with open(EXPORT_DIR / 'feature_cols.json') as f:
        model_state['feature_cols'] = json.load(f)
    print(f'Loaded exported model, expecting {len(model_state["feature_cols"])} features')

    if CHROMA_DIR.exists():
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        model_state['knowledge_base'] = chroma_client.get_or_create_collection(name='project_knowledge')
        print('Loaded RAG knowledge base')
    else:
        model_state['knowledge_base'] = None
        print('No chroma_data/ found — /ask will be unavailable until build_knowledge_base.py is run')

    yield
    model_state.clear()


app = FastAPI(title="CMAPSS RUL Prediction API", lifespan=lifespan)


class SensorReading(BaseModel):
    readings: Dict[str, float]


class Question(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in model_state}


@app.post("/predict")
def predict(reading: SensorReading):
    feature_cols = model_state['feature_cols']
    missing = [c for c in feature_cols if c not in reading.readings]
    if missing:
        raise HTTPException(status_code=400, detail=f'Missing sensor values: {missing}')

    row = pd.DataFrame([[reading.readings[c] for c in feature_cols]], columns=feature_cols)
    predicted_rul = model_state['model'].predict(row)[0]
    return {"predicted_rul_cycles": float(predicted_rul)}


def extract_sensor_mentions(text):
    matches = SENSOR_ID_PATTERN.findall(text)
    return [f'sensor_{m}' for m in matches]

@app.post("/ask")
def ask(question: Question):
    collection = model_state.get('knowledge_base')
    if collection is None:
        raise HTTPException(
            status_code=503,
            detail='Knowledge base not built — run build_knowledge_base.py first'
        )
    if not os.environ.get('ANTHROPIC_API_KEY'):
        raise HTTPException(
            status_code=500,
            detail='ANTHROPIC_API_KEY is not set — required for /ask'
        )

    # Semantic search: good for conceptual questions ("what was the best
    # RMSE?"), weak at distinguishing near-identical short strings that
    # differ mainly by a number (e.g. "sensor_11" vs "sensor_12").
    semantic_results = collection.query(query_texts=[question.question], n_results=RAG_TOP_K)
    doc_map = dict(zip(semantic_results['ids'][0], semantic_results['documents'][0]))

    # Exact-match lookup: if the question names a specific sensor by number,
    # fetch that document directly instead of relying on embedding similarity.
    exact_mentions = extract_sensor_mentions(question.question)
    for mention in exact_mentions:
        try:
            exact_results = collection.get(where_document={"$contains": mention})
            for doc_id, doc in zip(exact_results['ids'], exact_results['documents']):
                doc_map[doc_id] = doc
        except Exception:
            pass

    retrieved_ids = list(doc_map.keys())
    retrieved_docs = list(doc_map.values())

    context = '\n'.join(f'- {doc}' for doc in retrieved_docs)
    system_prompt = (
        "You are an assistant answering questions about a machine learning "
        "project that predicts turbofan engine Remaining Useful Life (RUL). "
        "Answer ONLY using the context below. If the context doesn't contain "
        "the answer, say you don't have that information — don't guess.\n\n"
        f"Context:\n{context}"
    )

    client = Anthropic()
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=500,
        system=system_prompt,
        messages=[{"role": "user", "content": question.question}],
    )

    return {
        "answer": message.content[0].text,
        "sources": retrieved_ids,
    }


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='127.0.0.1', port=8001)
