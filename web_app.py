"""
Servidor web para reconhecimento/cadastro pelo navegador (PC ou celular).

O GitHub não executa Python: você inicia este servidor no PC e abre o link
mostrado no terminal (rede local) ou um túnel HTTPS (ex.: ngrok) para a câmera
ao vivo no celular funcionar em mais navegadores.

Uso:
  uvicorn web_app:app --host 0.0.0.0 --port 8000

Depois abra no PC: http://127.0.0.1:8000
No celular (mesma Wi-Fi): http://<IP_DO_PC>:8000
"""

from __future__ import annotations

import socket
import sys
from contextlib import asynccontextmanager
from typing import Any

import cv2
import face_recognition
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from sistema_final_perfeito import FaceRecognitionSystem

_sistema: FaceRecognitionSystem | None = None


def _lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def _decode_image(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


def _find_faces(sistema: FaceRecognitionSystem, frame: np.ndarray) -> np.ndarray:
    faces = sistema.detect_faces(frame)
    if len(faces) > 0:
        return faces
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    m = max(h, w)
    scale = 640 / m if m > 640 else 1.0
    if scale < 1.0:
        small = cv2.resize(rgb, (int(w * scale), int(h * scale)))
    else:
        small = rgb
    locs = face_recognition.face_locations(small, model="hog")
    inv = 1.0 / scale if scale < 1.0 else 1.0
    out: list[tuple[int, int, int, int]] = []
    for top, right, bottom, left in locs:
        out.append(
            (
                int(left * inv),
                int(top * inv),
                int((right - left) * inv),
                int((bottom - top) * inv),
            )
        )
    return np.array(out) if out else np.array([])


def _largest_face(faces: np.ndarray) -> tuple[int, int, int, int] | None:
    if faces is None or len(faces) == 0:
        return None
    best = None
    best_a = 0
    for (x, y, w, h) in faces:
        a = w * h
        if a > best_a:
            best_a = a
            best = (int(x), int(y), int(w), int(h))
    return best


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _sistema
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    _sistema = FaceRecognitionSystem(headless=True)
    ip = _lan_ip()
    print("\n" + "=" * 60)
    print("  Facial Python — modo web")
    print("=" * 60)
    print(f"  PC:    http://127.0.0.1:8000")
    print(f"  Rede:  http://{ip}:8000  (celular na mesma Wi-Fi)")
    print("  Câmera ao vivo no celular costuma exigir HTTPS; use ngrok")
    print("  ou tire foto pelo botão «Tirar foto», que funciona em HTTP.")
    print("=" * 60 + "\n")
    yield
    if _sistema is not None:
        try:
            _sistema.cleanup()
        except Exception:
            if hasattr(_sistema, "conn") and _sistema.conn:
                _sistema.conn.close()


app = FastAPI(title="Facial Python Web", lifespan=lifespan)


@app.get("/api/status")
def api_status() -> dict[str, Any]:
    if _sistema is None:
        raise HTTPException(503, "Sistema não inicializado")
    return {
        "pessoas": len(_sistema.known_face_names),
        "nomes": list(_sistema.known_face_names),
    }


@app.post("/api/recognize")
async def api_recognize(image: UploadFile = File(...)) -> JSONResponse:
    if _sistema is None:
        raise HTTPException(503, "Sistema não inicializado")
    raw = await image.read()
    frame = _decode_image(raw)
    if frame is None:
        raise HTTPException(400, "Imagem inválida")
    faces = _find_faces(_sistema, frame)
    rect = _largest_face(faces)
    if rect is None:
        return JSONResponse(
            {"ok": True, "face_found": False, "name": None, "confidence": 0.0}
        )
    name, conf = _sistema.recognize_face_simple(frame, rect)
    if name:
        _sistema.log_recognition(name, float(conf))
    return JSONResponse(
        {
            "ok": True,
            "face_found": True,
            "name": name,
            "confidence": round(float(conf), 4),
            "box": {"x": rect[0], "y": rect[1], "w": rect[2], "h": rect[3]},
        }
    )


@app.post("/api/register")
async def api_register(
    image: UploadFile = File(...),
    nome: str = Form(...),
    amostra: bool = Form(False),
) -> JSONResponse:
    if _sistema is None:
        raise HTTPException(503, "Sistema não inicializado")
    raw = await image.read()
    frame = _decode_image(raw)
    if frame is None:
        raise HTTPException(400, "Imagem inválida")
    faces = _find_faces(_sistema, frame)
    rect = _largest_face(faces)
    if rect is None:
        raise HTTPException(400, "Nenhuma face detectada na imagem")
    ok, msg = _sistema.cadastrar_web(frame, rect, nome, adicionar_amostra=amostra)
    if not ok:
        raise HTTPException(400, msg)
    return JSONResponse({"ok": True, "message": msg})


INDEX_HTML = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Facial Python — Web</title>
  <style>
    :root { --bg:#0d1117; --card:#161b22; --text:#e6edf3; --accent:#58a6ff; --ok:#3fb950; --err:#f85149; }
    * { box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; background: var(--bg); color: var(--text);
      margin: 0; padding: 16px; max-width: 520px; margin-left: auto; margin-right: auto; }
    h1 { font-size: 1.25rem; margin: 0 0 8px; }
    p.hint { font-size: 0.85rem; opacity: 0.85; margin: 0 0 16px; line-height: 1.45; }
    .card { background: var(--card); border-radius: 12px; padding: 16px; margin-bottom: 16px;
      border: 1px solid #30363d; }
    video, canvas { width: 100%; border-radius: 8px; background: #000; display: block; }
    video { max-height: 280px; object-fit: cover; }
    .row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
    button, label.btn { cursor: pointer; border: 0; border-radius: 8px; padding: 10px 14px;
      font-size: 0.95rem; background: var(--accent); color: #fff; }
    label.btn { display: inline-block; background: #238636; }
    button.secondary { background: #30363d; color: var(--text); }
    input[type="text"] { width: 100%; padding: 10px; border-radius: 8px; border: 1px solid #30363d;
      background: #0d1117; color: var(--text); margin-top: 8px; }
    #out { margin-top: 12px; padding: 12px; border-radius: 8px; background: #21262d; font-size: 0.95rem; white-space: pre-wrap; }
    #out.err { border: 1px solid var(--err); }
    #out.ok { border: 1px solid var(--ok); }
    .hidden { display: none !important; }
    #statusLine { font-size: 0.8rem; opacity: 0.7; margin-bottom: 16px; }
  </style>
</head>
<body>
  <h1>Facial Python</h1>
  <p class="hint">Servidor rodando no seu PC. No celular, use a mesma rede Wi-Fi e o endereço
    que apareceu no terminal. «Tirar foto» funciona sem HTTPS; vídeo ao vivo pode exigir HTTPS (ex.: ngrok).</p>
  <p id="statusLine"></p>

  <div class="card">
    <strong>Pré-visualização</strong>
    <video id="vid" playsinline autoplay muted class="hidden"></video>
    <canvas id="cv" class="hidden"></canvas>
    <div class="row">
      <button type="button" id="btnCam">Usar câmera ao vivo</button>
      <button type="button" class="secondary" id="btnStop">Parar câmera</button>
    </div>
    <div class="row">
      <label class="btn">Tirar foto<input type="file" accept="image/*" capture="user" id="fileIn" class="hidden"></label>
    </div>
  </div>

  <div class="card">
    <strong>Reconhecer</strong>
    <div class="row">
      <button type="button" id="btnRec">Reconhecer (foto atual)</button>
    </div>
  </div>

  <div class="card">
    <strong>Cadastrar / amostra extra</strong>
    <input type="text" id="nome" placeholder="Nome da pessoa" autocomplete="name">
    <div class="row">
      <button type="button" id="btnCad">Cadastrar novo</button>
      <button type="button" class="secondary" id="btnAmostra">Só amostra extra</button>
    </div>
  </div>

  <div id="out"></div>

<script>
const vid = document.getElementById('vid');
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const out = document.getElementById('out');
const statusLine = document.getElementById('statusLine');

async function refreshStatus() {
  try {
    const r = await fetch('/api/status');
    const j = await r.json();
    statusLine.textContent = 'Cadastrados: ' + j.pessoas + (j.nomes.length ? ' — ' + j.nomes.join(', ') : '');
  } catch (e) {
    statusLine.textContent = 'Não foi possível ler /api/status';
  }
}
refreshStatus();

function setOut(text, ok) {
  out.textContent = text;
  out.className = ok ? 'ok' : 'err';
}

function frameBlob(cb) {
  if (vid.srcObject && vid.videoWidth > 0) {
    cv.width = vid.videoWidth;
    cv.height = vid.videoHeight;
    ctx.drawImage(vid, 0, 0);
    cv.toBlob(cb, 'image/jpeg', 0.85);
    return;
  }
  if (cv.width > 0 && cv.height > 0) {
    cv.toBlob(cb, 'image/jpeg', 0.85);
    return;
  }
  setOut('Ative a câmera ou use «Tirar foto».', false);
}

document.getElementById('btnCam').onclick = async () => {
  out.textContent = '';
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
    vid.srcObject = stream;
    vid.classList.remove('hidden');
    setOut('Câmera ativa. Use «Reconhecer» ou cadastro.', true);
  } catch (e) {
    setOut('Câmera bloqueada ou indisponível. Em HTTP no celular use «Tirar foto».', false);
  }
};

document.getElementById('btnStop').onclick = () => {
  const s = vid.srcObject;
  if (s) s.getTracks().forEach(t => t.stop());
  vid.srcObject = null;
  vid.classList.add('hidden');
};

document.getElementById('fileIn').onchange = async (ev) => {
  const f = ev.target.files && ev.target.files[0];
  if (!f) return;
  const url = URL.createObjectURL(f);
  const img = new Image();
  img.onload = () => {
    cv.width = img.width;
    cv.height = img.height;
    ctx.drawImage(img, 0, 0);
    URL.revokeObjectURL(url);
    cv.classList.remove('hidden');
    vid.classList.add('hidden');
    setOut('Foto carregada. Toque em Reconhecer ou cadastrar.', true);
  };
  img.src = url;
  ev.target.value = '';
};

function postImage(url, blob, extraForm) {
  const fd = new FormData();
  fd.append('image', blob, 'frame.jpg');
  if (extraForm) for (const k in extraForm) fd.append(k, extraForm[k]);
  return fetch(url, { method: 'POST', body: fd });
}

document.getElementById('btnRec').onclick = () => {
  frameBlob(async (blob) => {
    if (!blob) return;
    try {
      const r = await postImage('/api/recognize', blob);
      const j = await r.json();
      if (!r.ok) { setOut(j.detail || 'Erro', false); return; }
      if (!j.face_found) setOut('Nenhuma face detectada.', false);
      else if (j.name) setOut('Reconhecido: ' + j.name + ' (' + (j.confidence * 100).toFixed(1) + '% confiança)', true);
      else setOut('Face detectada, mas não cadastrada.', false);
    } catch (e) {
      setOut(String(e), false);
    }
  });
};

function doCad(amostra) {
  const nome = document.getElementById('nome').value.trim();
  if (!nome) { setOut('Preencha o nome.', false); return; }
  frameBlob(async (blob) => {
    if (!blob) return;
    try {
      const r = await postImage('/api/register', blob, { nome, amostra: amostra ? 'true' : 'false' });
      const j = await r.json();
      if (!r.ok) { setOut(typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail), false); return; }
      setOut(j.message || 'OK', true);
      refreshStatus();
    } catch (e) {
      setOut(String(e), false);
    }
  });
}

document.getElementById('btnCad').onclick = () => doCad(false);
document.getElementById('btnAmostra').onclick = () => doCad(true);
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML
