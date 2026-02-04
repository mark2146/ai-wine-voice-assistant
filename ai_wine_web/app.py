# app.py
from fastapi import FastAPI, UploadFile, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import io
from fastapi import Body
from dotenv import load_dotenv
load_dotenv()   # ⬅⬅⬅ 一定要在最前面
from urllib.parse import quote

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
async def chat(audio: UploadFile):
    wav_bytes = await audio.read()

    # STT
    user_text = speech_to_text(wav_bytes)
    print(f"[STT] user_text = {user_text!r}")

    if not user_text:
        print("[STT] empty text, return empty audio")
        return StreamingResponse(
            io.BytesIO(b""),
            media_type="audio/wav"
        )

    # RAG + GPT
    context = rag_search(user_text, index, texts)
    answer = ask_gpt(user_text, context)

    print(f"[GPT] answer = {answer!r}")

    # TTS
    tts_bytes = text_to_speech_wav_bytes(answer)

    headers = {
    "X-User-Text": quote(user_text),
    "X-AI-Text": quote(answer),
}

    return StreamingResponse(
        io.BytesIO(tts_bytes),
        media_type="audio/wav",
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