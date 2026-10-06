"""PillSense: medicine-leaflet assistant (Gemini Vision + RAG + Gradio)."""
import os, re, glob, json, time, logging, tempfile
import numpy as np
import gradio as gr
from google import genai
from utils import BASE, KNOWN, normalize_name, find_drugs_in_text

# ---------------- logging (monitoring) ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pillsense")

# ---------------- config (environment variables, never hardcoded) ----------------
API_KEY = os.environ.get("GEMINI_API_KEY")
TEXT_MODELS = os.environ.get(
    "TEXT_MODELS", "gemini-3.8-flash,gemini-flash-latest,gemini-flash-lite-latest").split(",")
EMB_MODEL = os.environ.get("EMB_MODEL", "gemini-embedding-001")
client = genai.Client(api_key=API_KEY) if API_KEY else None
if client is None:
    log.warning("GEMINI_API_KEY is not set: the app will only show leaflet excerpts.")

# models that failed (quota / not found) are skipped for a while instead of wasting requests
_cooldown = {}
COOLDOWN_SECONDS = 600


def call_gemini(contents, json_mode=False):
    """Try each model once (plus one retry on overload). Returns text or None."""
    if client is None:
        return None
    cfg = {"response_mime_type": "application/json"} if json_mode else None
    for model in TEXT_MODELS:
        if _cooldown.get(model, 0) > time.time():
            continue
        for attempt in range(2):
            try:
                return client.models.generate_content(
                    model=model, contents=contents, config=cfg).text
            except Exception as e:
                msg = str(e)
                if ("503" in msg or "UNAVAILABLE" in msg) and attempt == 0:
                    time.sleep(2)
                    continue
                log.warning(f"[{model}] skipped: {msg[:100]}")
                _cooldown[model] = time.time() + COOLDOWN_SECONDS
                break
    return None


# ---------------- Part 1: vision ----------------
_img_cache = {}


def get_drug_name(image):
    key = hash(image.tobytes())
    if key in _img_cache:
        return _img_cache[key]
    prompt = """Look at this photo of a medicine box, blister pack, or leaflet.
Return ONLY a JSON object with exactly these keys:
{"brand": "<brand name printed on the box, or unknown>",
 "generic": "<active ingredient in lowercase English, or unknown>"}
Do not add any other text."""
    raw = call_gemini([image, prompt], json_mode=True)
    try:
        data = json.loads(raw)
    except Exception:
        return "unknown"
    for k in ("generic", "brand"):
        val = (data.get(k) or "").strip()
        if val and val.lower() != "unknown":
            result = normalize_name(val)
            _img_cache[key] = result
            return result
    return "unknown"


# ---------------- Part 2: RAG ----------------
def load_chunks():
    chunks = []
    for path in sorted(glob.glob(os.path.join(BASE, "data", "*.txt"))):
        drug = os.path.splitext(os.path.basename(path))[0]
        text = open(path, encoding="utf-8").read()
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            line = re.sub(rf"^{drug}\.\s*", "", line, flags=re.I)
            m = re.match(r"^([A-Za-z ]+?):\s*(.*)$", line)
            section, body = (m.group(1), m.group(2)) if m else ("General", line)
            chunks.append({"drug": drug, "section": section,
                           "text": f"[{drug} | {section}] {body}"})
    return chunks


CHUNKS = load_chunks()
log.info(f"{len(CHUNKS)} chunks from {len({c['drug'] for c in CHUNKS})} drugs")


def embed(texts, batch=50):
    vecs = []
    for i in range(0, len(texts), batch):
        r = client.models.embed_content(model=EMB_MODEL, contents=texts[i:i + batch])
        vecs += [e.values for e in r.embeddings]
    arr = np.array(vecs, dtype="float32")
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


_EMB = None
_EMB_FILE = os.path.join(tempfile.gettempdir(), "pillsense_embeddings.npy")


def get_embeddings():
    """Document embeddings: loaded from a disk cache or computed once."""
    global _EMB
    if _EMB is not None:
        return _EMB
    if os.path.exists(_EMB_FILE):
        arr = np.load(_EMB_FILE)
        if arr.shape[0] == len(CHUNKS):
            _EMB = arr
            return _EMB
    if client is None:
        return None
    try:
        _EMB = embed([c["text"] for c in CHUNKS])
        np.save(_EMB_FILE, _EMB)
    except Exception as e:
        log.warning(f"embedding failed: {str(e)[:100]}")
    return _EMB


FALLBACK_ORDER = ["Interactions", "Warnings", "Who should not take it", "Uses", "How to take"]


def retrieve(question, drugs=None, k=3):
    emb = get_embeddings()
    q = None
    if emb is not None:
        try:
            q = embed([question])[0]
        except Exception as e:
            log.warning(f"query embedding failed: {str(e)[:100]}")
    results = []
    for d in (drugs if drugs else [None]):
        idx = [i for i, c in enumerate(CHUNKS) if d is None or c["drug"] == d]
        if q is not None:
            order = np.argsort(-(emb[idx] @ q))[:k]
            results += [CHUNKS[idx[j]] for j in order]
        else:  # no embeddings available: use the most safety-relevant sections
            idx.sort(key=lambda i: FALLBACK_ORDER.index(CHUNKS[i]["section"])
                     if CHUNKS[i]["section"] in FALLBACK_ORDER else 99)
            results += [CHUNKS[i] for i in idx[:k]]
    return results


# ---------------- Part 3: LLM ----------------
SYSTEM_RULES = """You are PillSense, a medicine leaflet assistant.
Rules:
1. Answer ONLY using the leaflet excerpts in CONTEXT. Do not use outside knowledge.
2. If the answer is not in CONTEXT, say it is not in the leaflet and advise asking a doctor or pharmacist.
3. Never invent doses. For children's doses, tell the user to check the product label or ask a doctor.
4. For interaction questions, check the Interactions section of EACH drug and state clearly whether a risk is mentioned.
5. Reply in the same language as the question (Egyptian Arabic if the question is Arabic). Keep it short and clear.
6. End with one short line: this is not a substitute for a doctor or pharmacist."""


def assistant(question, image=None, extra_drugs=None, k=3):
    t0 = time.time()
    drugs = []
    if image is not None:
        d = get_drug_name(image)
        if d != "unknown":
            drugs.append(d)
    for d in find_drugs_in_text(question) + [normalize_name(x) for x in (extra_drugs or [])]:
        if d not in drugs:
            drugs.append(d)

    unsupported = [d for d in drugs if d not in KNOWN]
    drugs = [d for d in drugs if d in KNOWN]

    if unsupported and not drugs:
        log.info(f"q={question!r} unsupported={unsupported}")
        return {"answer": f"الدوا ({', '.join(unsupported)}) مش موجود في قاعدة بيانات النشرات عندي. اسأل الصيدلي.",
                "drugs": [], "sources": []}

    if not drugs:  # no known medicine detected: don't search everything, refuse safely
        log.info(f"q={question!r} no_drug_detected")
        return {"answer": "مقدرتش أحدد دواء من قاعدة بياناتي. اكتب اسم الدواء أو ارفع صورة أوضح، أو اسأل الصيدلي.",
                "drugs": [], "sources": []}

    chunks = retrieve(question, drugs, k=k)
    context = "\n\n".join(c["text"] for c in chunks)
    prompt = f"{SYSTEM_RULES}\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
    answer = call_gemini(prompt)
    fallback = answer is None
    if fallback:  # graceful degradation: show the leaflet excerpts instead of an empty screen
        answer = ("⚠️ الموديل مشغول حاليًا. دي المقاطع ذات الصلة من النشرات:\n\n"
                  + "\n\n".join(f'**{c["drug"]} | {c["section"]}**\n{c["text"].split("] ", 1)[-1]}'
                                for c in chunks))
    if unsupported:
        answer += f"\n\n(ملحوظة: ({', '.join(unsupported)}) مش في قاعدة البيانات فمقدرتش أرد بخصوصه.)"

    log.info(f"q={question!r} drugs={drugs} image={image is not None} "
             f"fallback={fallback} latency={time.time() - t0:.1f}s")
    return {"answer": answer, "drugs": drugs,
            "sources": [f'{c["drug"]} | {c["section"]}' for c in chunks]}


# ---------------- UI ----------------
def pillsense_chat(message, image=None):
    if not message and image is None:
        return "اكتب سؤالك أو ارفع صورة الدواء.", "", ""
    try:
        result = assistant(
            question=message or "Identify this medicine and provide relevant information.",
            image=image)
        answer = result.get("answer", "مفيش إجابة متاحة.")
        drugs = result.get("drugs", [])
        sources = result.get("sources", [])
        drug_text = ", ".join(drugs) if drugs else "Unknown"
        source_text = "\n".join(f"• {s}" for s in sources) if sources else "No sources found."
        return answer, drug_text, source_text
    except Exception as e:
        log.exception("pipeline error")
        return f"حصل Error أثناء تشغيل النظام:\n{e}", "", ""


with gr.Blocks(title="PillSense - AI Medicine Assistant") as demo:
    gr.Markdown("""
    # 💊 PillSense
    ### AI-Powered Medicine Leaflet Assistant

    Ask questions about medicines, upload a medicine image,
    and get answers based on medicine leaflet information.
    """)
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="📷 Upload Medicine Image")
            question_input = gr.Textbox(label="💬 Your Question", lines=4,
                                        placeholder="Example: ينفع آخد بروفين مع وارفارين؟")
            with gr.Row():
                ask_btn = gr.Button("🔍 Ask PillSense", variant="primary")
                clear_btn = gr.Button("🗑️ Clear")
        with gr.Column(scale=1):
            answer_output = gr.Markdown(label="🤖 AI Answer")
            drug_output = gr.Textbox(label="💊 Detected Medicine", interactive=False)
            sources_output = gr.Textbox(label="📚 Sources Used", lines=6, interactive=False)
    gr.Markdown("---\n⚠️ **Disclaimer:** PillSense is an educational assistant and "
                "is not a substitute for a doctor or pharmacist. The knowledge base contains "
                "simplified educational summaries, not official leaflets.")

    ask_btn.click(pillsense_chat, [question_input, image_input],
                  [answer_output, drug_output, sources_output])
    clear_btn.click(lambda: ("", None, "", "", ""), [],
                    [question_input, image_input, answer_output, drug_output, sources_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
