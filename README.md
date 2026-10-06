---
title: PillSense
emoji: 💊
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
---

# 💊 PillSense: Medicine Leaflet Assistant

An AI assistant that explains medicines in simple English or Arabic, with answers grounded **only** in leaflet text.

## Problem
Medicine leaflets are long and full of jargon, and dangerous drug interactions are easy to miss.

## How it works
```
Photo / text → Gemini Vision reads the box → name normalized to generic (Panadol → paracetamol)
→ RAG retrieves matching leaflet chunks (gemini-embedding-001, cosine similarity)
→ Gemini answers using ONLY that context → answer + cited sources
```

## Features
- Recognizes medicines from a photo, brand name, Arabic name, or a typo
- Leaflet-grounded answers with cited sources (drug | section)
- Interaction questions across several drugs
- Safe refusal when a drug is not in the database
- Graceful fallback: if the LLM is unavailable, the app shows the raw leaflet excerpts

## Project structure
```
app.py            Gradio app: vision, RAG, LLM, logging
utils.py          normalize_name, find_drugs_in_text (unit-tested)
data/             10 simplified leaflet summaries (.txt)
config/           brands.json (brand / Arabic names → generic)
tests/            pytest unit tests
.github/          CI: runs tests on every push
Dockerfile        reproducible container
PillSense.ipynb   original Colab notebook
```

## Run locally
```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key      # never commit your key
python app.py                       # open http://localhost:7860
```

## Run with Docker
```bash
docker build -t pillsense .
docker run -p 7860:7860 -e GEMINI_API_KEY=your_key pillsense
```

## Tests
```bash
pip install pytest && pytest tests/
```

## Team
- Name 1: Vision & Data
- Name 2: RAG Engine
- Name 3: LLM & UI

## Disclaimer
Educational project. Not a substitute for a doctor or pharmacist.
The knowledge base contains simplified educational summaries, not official leaflets.
