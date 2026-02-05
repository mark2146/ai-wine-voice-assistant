from fastapi import FastAPI, UploadFile, Request, File, Body
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from urllib.parse import quote
import io

load_dotenv()

from ai_wine_web.rag import pipeline_tts_stream
from ai_wine_web.rag import (
    build_index,
    speech_to_text,
    rag_search,
    ask_gpt,
    text_to_speech_wav_bytes,
)

# =========================================================
# App init
# =========================================================
app = FastAPI()

# 載入 RAG index（只做一次）
index, texts = build_index()

# =========================================================
# Static / Frontend
# =========================================================
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    """
    前端入口頁
    只顯示 UI，不會自動啟動任何錄音或 API
    """
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()


# =========================================================
# Chat API
# =========================================================
@app.post("/chat")
async def chat(audio: UploadFile = File(...)):

    # ⭐ 讀取音訊
    wav_bytes = await audio.read()

    # ⭐ STT（直接吃 bytes）
    user_text = speech_to_text(wav_bytes)

    # ⭐ 防呆
    if not user_text:
        return StreamingResponse(
            iter([b""]),
            media_type="audio/mpeg"
        )

    # ⭐ RAG + GPT
    context = rag_search(user_text, index, texts, k=2)
    answer = ask_gpt(user_text, context)

    # ⭐ header 傳文字給前端
    headers = {
        "X-User-Text": quote(user_text),
        "X-AI-Text": quote(answer)
    }

    # ⭐ Pipeline TTS Streaming
    def audio_generator():
        for chunk in pipeline_tts_stream(answer):
            yield chunk

    return StreamingResponse(
        audio_generator(),
        media_type="audio/mpeg",
        headers=headers
    )
        
@app.post("/tts")
async def tts(payload: dict = Body(...)):
    text = payload.get("text", "").strip()
    if not text:
        return Response(status_code=400)

    audio_bytes = text_to_speech_wav_bytes(text)
    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type="audio/wav"
    )