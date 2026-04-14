"""
Servidor web — mesmo cadastro/reconhecimento do sistema desktop (face_recognition + .pkl).

Uso no PC e celular (mesma Wi-Fi):
  python -m uvicorn web_app:app --host 0.0.0.0 --port 8000

Importante: use --host 0.0.0.0 (não só 127.0.0.1), senão o celular não conecta.
"""

from __future__ import annotations

import io
import socket
import sys
import threading
from contextlib import asynccontextmanager
from typing import Any

import cv2
import face_recognition
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image, ImageOps

from sistema_final_perfeito import FaceRecognitionSystem

_sistema: FaceRecognitionSystem | None = None
_sistema_lock = threading.Lock()
_MAX_SIDE = 1600


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
    """BGR OpenCV; corrige rotação EXIF (fotos de celular)."""
    if not data:
        return None
    try:
        pil = Image.open(io.BytesIO(data))
        pil = ImageOps.exif_transpose(pil)
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        rgb = np.array(pil)
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return None
    h, w = frame.shape[:2]
    m = max(h, w)
    if m > _MAX_SIDE:
        s = _MAX_SIDE / m
        frame = cv2.resize(
            frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA
        )
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


def _form_bool_amostra(raw: str) -> bool:
    """Evita bool('false') == True em multipart/form-data."""
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


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
    print(f"  PC:     http://127.0.0.1:8000")
    print(f"  Celular (mesma Wi-Fi): http://{ip}:8000")
    print("  O servidor PRECISA usar --host 0.0.0.0 para o celular entrar.")
    print("=" * 60 + "\n")
    yield
    if _sistema is not None:
        try:
            _sistema.cleanup()
        except Exception:
            if hasattr(_sistema, "conn") and _sistema.conn:
                _sistema.conn.close()


app = FastAPI(title="Facial Python Web", lifespan=lifespan)


@app.get("/api/server")
def api_server(request: Request) -> dict[str, Any]:
    """URLs para abrir no PC e no celular."""
    ip = _lan_ip()
    port = request.url.port or 8000
    return {
        "lan_ip": ip,
        "url_pc": f"http://127.0.0.1:{port}",
        "url_celular": f"http://{ip}:{port}",
        "voce_esta_em": str(request.base_url).rstrip("/"),
    }


@app.get("/api/status")
def api_status() -> dict[str, Any]:
    if _sistema is None:
        raise HTTPException(503, "Sistema não inicializado")
    with _sistema_lock:
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
        raise HTTPException(400, "Imagem inválida ou vazia")
    with _sistema_lock:
        faces = _find_faces(_sistema, frame)
        rect = _largest_face(faces)
        if rect is None:
            return JSONResponse(
                {
                    "ok": True,
                    "face_found": False,
                    "name": None,
                    "confidence": 0.0,
                    "message": "Nenhuma face detectada. Tente outro ângulo ou mais luz.",
                }
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
                "message": (
                    f"Reconhecido: {name}"
                    if name
                    else "Face vista, mas não bate com ninguém cadastrado."
                ),
                "box": {"x": rect[0], "y": rect[1], "w": rect[2], "h": rect[3]},
            }
        )


@app.post("/api/register")
async def api_register(
    image: UploadFile = File(...),
    nome: str = Form(...),
    amostra: str = Form("false"),
) -> JSONResponse:
    if _sistema is None:
        raise HTTPException(503, "Sistema não inicializado")
    raw = await image.read()
    frame = _decode_image(raw)
    if frame is None:
        raise HTTPException(400, "Imagem inválida ou vazia")
    adicionar_amostra = _form_bool_amostra(amostra)
    nome = (nome or "").strip()
    with _sistema_lock:
        faces = _find_faces(_sistema, frame)
        rect = _largest_face(faces)
        if rect is None:
            raise HTTPException(
                400,
                "Nenhuma face detectada. Centralize o rosto, boa luz, e tente de novo.",
            )
        ok, msg = _sistema.cadastrar_web(
            frame, rect, nome, adicionar_amostra=adicionar_amostra
        )
    if not ok:
        raise HTTPException(400, msg)
    return JSONResponse({"ok": True, "message": msg})


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
  <title>Facial Python</title>
  <style>
    :root { --bg:#0d1117; --card:#161b22; --text:#e6edf3; --accent:#238636; --blue:#1f6feb; --err:#f85149; }
    * { box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; background: var(--bg); color: var(--text);
      margin: 0; padding: 12px; max-width: 560px; margin-left: auto; margin-right: auto; }
    h1 { font-size: 1.35rem; margin: 0 0 4px; }
    .links { background: #21262d; border: 1px solid #30363d; border-radius: 10px; padding: 12px; margin: 12px 0; font-size: 0.9rem; }
    .links a { color: #58a6ff; word-break: break-all; }
    .warn { color: #d29922; font-size: 0.85rem; margin: 8px 0; line-height: 1.4; }
    .card { background: var(--card); border-radius: 12px; padding: 14px; margin-bottom: 12px; border: 1px solid #30363d; }
    .card strong { display: block; margin-bottom: 8px; font-size: 1rem; }
    video, canvas { width: 100%; border-radius: 10px; background: #000; display: block; min-height: 120px; }
    video { max-height: 320px; object-fit: cover; }
    .row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
    .btn-big { flex: 1 1 100%; padding: 16px; font-size: 1.05rem; border: 0; border-radius: 10px;
      cursor: pointer; font-weight: 600; color: #fff; }
    .btn-rec { background: var(--blue); }
    .btn-cad { background: var(--accent); }
    .btn-sec { background: #30363d; color: var(--text); padding: 12px 14px; border: 0; border-radius: 8px; cursor: pointer; }
    label.filelab { display: block; text-align: center; padding: 14px; background: #30363d; border-radius: 10px;
      cursor: pointer; font-weight: 600; }
    label.filelab input { display: none; }
    input[type="text"] { width: 100%; padding: 14px; border-radius: 10px; border: 1px solid #30363d;
      background: #0d1117; color: var(--text); font-size: 1rem; margin-top: 6px; }
    #out { margin-top: 12px; padding: 14px; border-radius: 10px; background: #21262d; font-size: 0.95rem;
      white-space: pre-wrap; line-height: 1.45; border: 1px solid #30363d; }
    #out.err { border-color: var(--err); }
    #out.ok { border-color: #238636; }
    .hidden { display: none !important; }
    #statusLine { font-size: 0.82rem; opacity: 0.85; margin-bottom: 8px; }
  </style>
</head>
<body>
  <h1>Facial Python</h1>
  <p id="statusLine">Carregando…</p>
  <div class="links" id="linkBox">
    <div><strong>Abrir no celular</strong> (mesma Wi-Fi que o PC):</div>
    <div id="urlCel"></div>
    <div style="margin-top:8px"><strong>No PC:</strong> <a id="urlPc" href="#">127.0.0.1</a></div>
  </div>
  <p class="warn">Se o celular não abrir: confirme que o servidor foi iniciado com <code>--host 0.0.0.0</code>,
    desative “isolamento de cliente” no roteador se existir, e permita o Python no firewall do Windows.</p>

  <div class="card">
    <strong>1 — Foto do rosto</strong>
    <video id="vid" playsinline autoplay muted class="hidden"></video>
    <canvas id="cv" class="hidden"></canvas>
    <div class="row">
      <label class="filelab">📷 Tirar / escolher foto<input type="file" accept="image/*" capture="user" id="fileIn"></label>
    </div>
    <div class="row">
      <button type="button" class="btn-sec" id="btnCam">Câmera ao vivo (PC ou HTTPS)</button>
      <button type="button" class="btn-sec" id="btnStop">Parar câmera</button>
    </div>
  </div>

  <div class="card">
    <strong>2 — Reconhecer</strong>
    <span style="font-size:0.85rem;opacity:0.9">Igual à tecla ESPAÇO no programa antigo: usa a foto acima.</span>
    <div class="row">
      <button type="button" class="btn-big btn-rec" id="btnRec">RECONHECER</button>
    </div>
  </div>

  <div class="card">
    <strong>3 — Cadastrar novo</strong>
    <span style="font-size:0.85rem;opacity:0.9">Igual ao fluxo «C» no desktop: nome + face na foto.</span>
    <input type="text" id="nome" placeholder="Nome da pessoa" autocomplete="name" enterkeyhint="done">
    <div class="row">
      <button type="button" class="btn-big btn-cad" id="btnCad">CADASTRAR</button>
    </div>
    <div class="row">
      <button type="button" class="btn-sec" id="btnAmostra">Só amostra extra (já cadastrado)</button>
    </div>
  </div>

  <div id="out"></div>

<script>
const vid = document.getElementById('vid');
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const out = document.getElementById('out');
const statusLine = document.getElementById('statusLine');

function errText(j) {
  if (!j || j.detail === undefined) return 'Erro na resposta';
  const d = j.detail;
  if (typeof d === 'string') return d;
  if (Array.isArray(d)) return d.map(x => x.msg || JSON.stringify(x)).join('\n');
  return JSON.stringify(d);
}

async function refreshStatus() {
  try {
    const r = await fetch('/api/status');
    if (!r.ok) throw new Error();
    const j = await r.json();
    statusLine.textContent = 'Cadastrados: ' + j.pessoas + (j.nomes.length ? ' — ' + j.nomes.join(', ') : '');
  } catch (e) {
    statusLine.textContent = 'Sem conexão com o servidor. Confira o endereço e se o terminal está rodando uvicorn.';
  }
}

async function refreshLinks() {
  try {
    const r = await fetch('/api/server');
    const j = await r.json();
    const a = document.createElement('a');
    a.href = j.url_celular;
    a.textContent = j.url_celular;
    const box = document.getElementById('urlCel');
    box.innerHTML = '';
    box.appendChild(a);
    const pc = document.getElementById('urlPc');
    pc.href = j.url_pc;
    pc.textContent = j.url_pc;
  } catch (e) {}
}
refreshStatus();
refreshLinks();
setInterval(refreshStatus, 15000);

function setOut(text, ok) {
  out.textContent = text;
  out.className = ok ? 'ok' : 'err';
}

function frameBlob(cb) {
  if (vid.srcObject && vid.videoWidth > 0) {
    cv.width = vid.videoWidth;
    cv.height = vid.videoHeight;
    ctx.drawImage(vid, 0, 0);
    cv.classList.remove('hidden');
    cv.toBlob(cb, 'image/jpeg', 0.92);
    return;
  }
  if (cv.width > 0 && cv.height > 0) {
    cv.toBlob(cb, 'image/jpeg', 0.92);
    return;
  }
  setOut('Primeiro use «Tirar / escolher foto» ou a câmera ao vivo.', false);
}

document.getElementById('btnCam').onclick = async () => {
  out.textContent = '';
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
    vid.srcObject = stream;
    vid.classList.remove('hidden');
    setOut('Câmera ativa. Toque em RECONHECER ou CADASTRAR.', true);
  } catch (e) {
    setOut('Câmera ao vivo indisponível neste modo. Use «Tirar / escolher foto» (funciona no celular em HTTP).', false);
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
    setOut('Foto pronta. Toque em RECONHECER ou preencha o nome e CADASTRAR.', true);
  };
  img.onerror = () => { setOut('Não foi possível ler a imagem.', false); URL.revokeObjectURL(url); };
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
    setOut('Analisando…', true);
    try {
      const r = await postImage('/api/recognize', blob);
      const j = await r.json();
      if (!r.ok) { setOut(errText(j), false); return; }
      if (j.message) setOut(j.message, !!j.name);
      else if (!j.face_found) setOut('Nenhuma face detectada.', false);
      else if (j.name) setOut('Reconhecido: ' + j.name + ' (~' + (j.confidence * 100).toFixed(0) + '%)', true);
      else setOut('Face detectada, mas não cadastrada.', false);
    } catch (e) {
      setOut('Falha de rede. No celular use o link http://IP:8000 (mesma Wi-Fi) e servidor com --host 0.0.0.0.', false);
    }
  });
};

function doCad(amostra) {
  const nome = document.getElementById('nome').value.trim();
  if (!nome) { setOut('Digite o nome antes de cadastrar.', false); return; }
  frameBlob(async (blob) => {
    if (!blob) return;
    setOut('Cadastrando…', true);
    try {
      const r = await postImage('/api/register', blob, { nome, amostra: amostra ? 'true' : 'false' });
      const j = await r.json();
      if (!r.ok) { setOut(errText(j), false); return; }
      setOut(j.message || 'Cadastro OK', true);
      refreshStatus();
    } catch (e) {
      setOut('Falha de rede. Confira Wi-Fi, firewall e se o servidor usa --host 0.0.0.0.', false);
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
