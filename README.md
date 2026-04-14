# Facial Python

**English:** Face recognition system with a desktop HUD (OpenCV) and a **browser UI** (FastAPI) for PC and mobile on the same Wi‑Fi. Uses `face_recognition` (dlib 128‑D embeddings), SQLite for logs, and pickle for registered encodings.

**Português:** Sistema de **reconhecimento facial** com interface desktop estilo HUD (OpenCV) e **versão web** (FastAPI) para uso no navegador no PC ou celular (rede local). Utiliza `face_recognition` (embeddings 128‑D via dlib), SQLite para histórico e pickle para codificações cadastradas.

---

## Funcionalidades

| Modo | Descrição |
|------|-----------|
| **Desktop** (`sistema_final_perfeito.py`) | Webcam, HUD futurista, cadastro (tecla `c`), reconhecimento (`espaço`), amostras extras, exportação CSV, estatísticas. |
| **Web** (`web_app.py`) | Cadastro e reconhecimento pelo navegador; fotos do celular com correção EXIF; mesmo modelo de dados do desktop. |

---

## Stack

Python · OpenCV · NumPy · Pillow · [face_recognition](https://github.com/ageitgey/face_recognition) · dlib · SQLite · FastAPI · Uvicorn

---

## Estrutura do repositório

```
facial_python/
├── sistema_final_perfeito.py   # Núcleo: detecção Haar, encodings, HUD, SQLite
├── web_app.py                  # API + página HTML (cadastro / reconhecer)
├── run_web.bat                 # Windows: sobe o servidor com host 0.0.0.0
├── requirements.txt
├── README.md
└── LICENSE
```

Arquivos de **dados locais** (encodings, banco, fotos) ficam fora do Git por privacidade e porte — veja `.gitignore`.

---

## Requisitos

- **Python 3.10+** (testado em 3.13 no Windows)
- **Windows:** Visual C++ Build Tools podem ser necessários para compilar **dlib** se não houver wheel compatível.
- Webcam (somente modo desktop).

---

## Instalação

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
```

---

## Como executar

### Interface web (PC ou celular na mesma rede)

```bash
python -m uvicorn web_app:app --host 0.0.0.0 --port 8000
```

No Windows você pode usar **`run_web.bat`**. Abra no navegador:

- PC: `http://127.0.0.1:8000`
- Celular: `http://<IP_DO_PC>:8000` (o próprio site mostra o link quando possível)

> Use sempre `--host 0.0.0.0` se for acessar pelo celular. Com `127.0.0.1` só o próprio computador conecta.

### Interface desktop (HUD)

```bash
python sistema_final_perfeito.py
```

---

## API web (referência rápida)

| Método | Rota | Uso |
|--------|------|-----|
| `GET` | `/` | Interface web |
| `GET` | `/api/status` | Quantidade e nomes cadastrados |
| `GET` | `/api/server` | URLs sugeridas (PC / rede local) |
| `POST` | `/api/recognize` | `multipart/form-data`: campo `image` (JPEG) |
| `POST` | `/api/register` | Campos `image`, `nome`, opcional `amostra` (`true` / `false`) |

---

## Privacidade e dados

Encodings (`.pkl`), banco (`.db`) e pastas de fotos/exportações **não são versionados**. Quem clonar o repositório começa sem cadastros e gera os arquivos localmente após uso.

---

## Licença

MIT — veja [LICENSE](LICENSE).

---

## Autor

**Renan Gomes** — [github.com/RenanGomes01/facial_python](https://github.com/RenanGomes01/facial_python)
