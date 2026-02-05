let audioContext;
let processor;
let input;
let stream;

let pcmData = [];
let recording = false;

// LOCKED → ARMED → SESSION
let systemState = "LOCKED";

const SAMPLE_RATE = 16000;
const SILENCE_THRESHOLD = 0.008;
const SILENCE_TIMEOUT = 2500;

const startBtn = document.getElementById("startBtn");
const statusEl = document.getElementById("status");
const player = document.getElementById("player");

/* =========================
   WAV ENCODER
========================= */
function floatTo16BitPCM(float32Array) {
  const buffer = new ArrayBuffer(float32Array.length * 2);
  const view = new DataView(buffer);
  let offset = 0;

  for (let i = 0; i < float32Array.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, float32Array[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buffer;
}

function encodeWAV(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  function writeString(o, s) {
    for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i));
  }

  writeString(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, samples.length * 2, true);

  new Uint8Array(buffer, 44).set(new Uint8Array(floatTo16BitPCM(samples)));
  return buffer;
}

/* =========================
   RECORD WITH SILENCE DETECT
========================= */
async function recordWithSilenceDetect() {
  console.log("[REC] start recording");

  pcmData = [];
  recording = true;

  stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
  input = audioContext.createMediaStreamSource(stream);

  processor = audioContext.createScriptProcessor(4096, 1, 1);
  input.connect(processor);
  processor.connect(audioContext.destination);

  let silenceStart = null;
  let hasSpoken = false;

  return new Promise(resolve => {
    processor.onaudioprocess = e => {
      if (!recording) return;

      const data = e.inputBuffer.getChannelData(0);
      pcmData.push(new Float32Array(data));

      const rms = Math.sqrt(
        data.reduce((s, x) => s + x * x, 0) / data.length
      );

      // ★ 關鍵：提高 threshold，避免底噪
      if (rms > 0.015) {
        hasSpoken = true;
        silenceStart = null;
      } else if (hasSpoken) {
        if (!silenceStart) silenceStart = performance.now();
        if (performance.now() - silenceStart > 1200) {
          stopRecording(resolve, true);
        }
      }
    };

    // ★ 硬上限，避免無限錄音
    setTimeout(() => stopRecording(resolve, hasSpoken), 6000);
  });
}

function stopRecording(resolve, hasSpoken) {
  recording = false;

  try { processor.disconnect(); } catch {}
  try { input.disconnect(); } catch {}
  try { stream.getTracks().forEach(t => t.stop()); } catch {}
  try { audioContext.close(); } catch {}

  if (!hasSpoken || pcmData.length === 0) {
    resolve(null);
    return;
  }

  let len = pcmData.reduce((s, a) => s + a.length, 0);
  let samples = new Float32Array(len);
  let off = 0;
  for (let a of pcmData) {
    samples.set(a, off);
    off += a.length;
  }
  resolve(samples);
}

/* =========================
   BACKEND COMM
========================= */
async function sendAudio(samples) {

  const wav = encodeWAV(samples, SAMPLE_RATE);
  const blob = new Blob([wav], { type: "audio/wav" });

  const fd = new FormData();
  fd.append("audio", blob, "speech.wav");

  const res = await fetch("/chat", { method: "POST", body: fd });
  if (!res.ok) throw new Error(await res.text());

  // header文字保持
  const userText = decodeURIComponent(res.headers.get("X-User-Text") || "");
  const aiText = decodeURIComponent(res.headers.get("X-AI-Text") || "");

  if (userText) appendChat("你", userText);
  if (aiText) appendChat("AI", aiText);

  // ⭐ 直接串流播放
  await playStream(res);
}
function appendChat(role, text) {
  const log = document.getElementById("chatLog");
  if (!log) return;

  const div = document.createElement("div");
  div.className = "chat__item";
  div.innerHTML = `<b>${role}：</b> ${text.replace(/\n/g, "<br>")}`;

  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}
async function sendTextTTS(text) {
  const res = await fetch("/tts", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text })
  });
  if (!res.ok) throw new Error(await res.text());
  return await res.blob();
}

function playAudio(blob) {
  if (!blob || blob.size < 1000) {
    console.error("❌ Invalid audio blob");
    return Promise.resolve();
  }

  return new Promise(resolve => {
    player.src = URL.createObjectURL(blob);
    player.onended = resolve;
    player.onerror = () => resolve();
    player.play().catch(() => resolve());
  });
}
async function playStream(response) {

  const reader = response.body.getReader();

  const mediaSource = new MediaSource();
  player.src = URL.createObjectURL(mediaSource);

  await new Promise(resolve => {

    mediaSource.addEventListener("sourceopen", async () => {

      const sourceBuffer = mediaSource.addSourceBuffer("audio/mpeg");

      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          mediaSource.endOfStream();
          resolve();
          break;
        }

        await new Promise(r => {
          sourceBuffer.addEventListener("updateend", r, { once: true });
          sourceBuffer.appendBuffer(value);
        });
      }

    });

  });

  await player.play();
}

async function startSession() {
  systemState = "SESSION";
  let noSpeechCount = 0;

  statusEl.textContent = "🤖 系統啟動中…";
  await playAudio(await sendTextTTS("歡迎光臨 AI 紅酒櫃，請問需要什麼協助？"));
  await sleep(1500);

  while (systemState === "SESSION") {
    statusEl.textContent = "🎧 收音中…";
    await sleep(300);

    const audio = await recordWithSilenceDetect();

    if (!audio) {
      noSpeechCount++;
      if (noSpeechCount < 2) {
        statusEl.textContent = "⏳ 沒聽到聲音，請再試一次…";
        await sleep(1000);
        continue;
      }

      statusEl.textContent = "👋 結束互動中…";
      await playAudio(await sendTextTTS("謝謝光臨，有需要可以再叫我"));
      setArmed();
      break;
    }

    noSpeechCount = 0;

    statusEl.textContent = "🤖 思考中…";
    await sendAudio(audio);


    // ★ 非常重要：播放完後冷卻
    statusEl.textContent = "⏸ 等待中…";
    await sleep(1200);
  }
}
/* =========================
   UI / UTILS
========================= */
function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

function setLocked() {
  systemState = "LOCKED";
  startBtn.disabled = false;
  statusEl.textContent = "尚未啟動";
}

function setArmed() {
  systemState = "ARMED";
  startBtn.disabled = true;
  statusEl.textContent = "⏸ 待命中：按 Enter 啟動";
}

startBtn.onclick = () => {
  if (systemState === "LOCKED") setArmed();
};

document.addEventListener("keydown", e => {
  if (e.key === "Enter" && systemState === "ARMED") {
    startSession();
  }
});

setLocked();
