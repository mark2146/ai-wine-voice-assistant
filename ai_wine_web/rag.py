# rag.py
import io, os, json, glob
import numpy as np
import soundfile as sf
import faiss
from openai import OpenAI
import re

# -------------------------
# API KEY（只允許 env）
# -------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------
# Models
# -------------------------
CHAT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
STT_MODEL = "gpt-4o-transcribe"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "alloy"
# alloy      → 中性 AI 聲
# verse      → 比較自然敘事感
# aria       → 偏女性
# sage       → 偏成熟沉穩

# -------------------------
# Paths
# -------------------------
BASE_DIR = os.path.dirname(__file__)
KB_DIR = os.path.join(BASE_DIR, "kb")
INDEX_PATH = os.path.join(BASE_DIR, "kb.index")
META_PATH = os.path.join(BASE_DIR, "kb_texts.json")

class StreamingTextChunker:

    def __init__(self, min_chars=20):
        self.buffer = ""
        self.min_chars = min_chars

    def push(self, text):

        self.buffer += text

        # 優先找中文停頓符
        split_chars = "，。！？,.!? "

        if len(self.buffer) < self.min_chars:
            return None

        for i, ch in enumerate(self.buffer):
            if ch in split_chars and i >= self.min_chars:
                chunk = self.buffer[:i+1]
                self.buffer = self.buffer[i+1:]
                return chunk

        return None

    def flush(self):
        if self.buffer:
            chunk = self.buffer
            self.buffer = ""
            return chunk
        return None
    
def pipeline_tts_stream(text):

    chunker = StreamingTextChunker()

    # 模擬 GPT streaming（目前先用整段文字）
    for token in text:

        chunk = chunker.push(token)

        if chunk:
            print("🎤 TTS chunk:", chunk)

            audio = text_to_speech_wav_bytes(chunk)
            yield audio

    # flush
    last = chunker.flush()
    if last:
        yield text_to_speech_wav_bytes(last)
   
# -------------------------
# STT
# -------------------------
def speech_to_text(wav_bytes: bytes) -> str:
    try:
        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    except Exception as e:
        raise RuntimeError(f"Invalid audio input: {e}")

    # mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # resample if needed
    if sr != 16000:
        try:
            import resampy
            data = resampy.resample(data, sr, 16000)
            sr = 16000
        except ImportError:
            # fallback：不 resample，至少不炸
            pass

    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    buf.name = "speech.wav"

    result = client.audio.transcriptions.create(
        model=STT_MODEL,
        file=buf
    )
    return (result.text or "").strip()

# -------------------------
# TTS
# -------------------------
def text_to_speech_wav_bytes(text: str) -> bytes:
    audio_response = client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text,
        speed=1.05
    )
    return audio_response.read()

# -------------------------
# KB
# -------------------------
def load_docs(chunk_size=300):
    docs = []
    for path in glob.glob(os.path.join(KB_DIR, "*.txt")):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        for p in text.split("\n\n"):
            p = p.strip()
            if p:
                docs.append(p[:chunk_size])
    return docs

def build_index():
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        index = faiss.read_index(INDEX_PATH)
        texts = json.load(open(META_PATH, encoding="utf-8"))
        return index, texts

    texts = load_docs()
    if not texts:
        raise RuntimeError(f"KB 資料夾 '{KB_DIR}' 裡沒有任何 .txt 文件")

    emb = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vectors = np.array([e.embedding for e in emb.data], dtype=np.float32)
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, INDEX_PATH)
    json.dump(texts, open(META_PATH, "w", encoding="utf-8"), ensure_ascii=False)

    return index, texts

def rag_search(query, index, texts, k=2):
    q_emb = client.embeddings.create(model=EMBED_MODEL, input=[query])
    qv = np.array(q_emb.data[0].embedding, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(qv)
    _, ids = index.search(qv, k)
    return "\n\n".join(texts[i] for i in ids[0] if i >= 0)

# -------------------------
# GPT
# -------------------------
def ask_gpt(question, context):
    system_prompt = (
        "你是 AI 紅酒櫃語音助理，專門介紹全家的葡萄酒。"
        "請用繁體中文，但酒名用英文，語氣專業親切，適合語音播放。"
    )

    resp = client.responses.create(
        model=CHAT_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{question}\n\n{context}"}
        ]
    )
    return (resp.output_text or "").strip()
