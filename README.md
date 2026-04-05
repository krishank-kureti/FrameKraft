# FrameKraft 🎨🎵

> **Transform any image into a stylized visual experience with a matching soundtrack.**

FrameKraft is an AI-powered web application that takes a single photo and runs it through a creative pipeline. It generates captions, applies artistic edits, and curates music that matches the mood — all in one seamless flow.

---

## ✨ What does it do?

FrameKraft runs a **5-stage AI pipeline** on your image:

1. 📸 **BLIP Captioning** — Generates descriptive captions using `Salesforce/blip-image-captioning-base`.
2. 🏷️ **CLIP Classification** — Validates and ranks captions via `openai/clip-vit-base-patch32`.
3. ✍️ **Groq Stylization** — Creates 3 mood-based captions and picks an editing vibe using `llama-3.3-70b-versatile`.
4. 🖌️ **OpenCV Editing** — Applies smart filters (brightness, temperature, vignette, etc.) based on your chosen vibe.
5. 🎵 **Music RAG** — Finds matching songs from a 10K database and fetches iTunes 30-second previews.

---

## 🛠️ Prerequisites

- **Python 3.10+** 🐍
- **[Groq API Key](https://console.groq.com/)** 🔑 *(required for LLM calls)*
- *(Optional)* **GPU** — speeds up model inference, but CPU works fine too 🚀

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd FrameKraft
   ```

2. **Set up the virtual environment**
   ```bash
   cd backend
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   > 💡 *First run will download ~1GB of model weights (BLIP & CLIP). Subsequent runs use the cache.*

---

## ⚙️ Configuration

Create a `.env` file inside the `backend/` directory:

```env
GROQ_API_KEY=your-groq-api-key-here
```

> 🔒 **Never commit your `.env` file to version control.**

---

## 🌐 Running the Web App

Start the FastAPI server:

```bash
uvicorn app.main:app --reload --port 8000
```

Open your browser and visit: **[http://localhost:8000](http://localhost:8000)** 🌍

### 📱 Web UI Flow
1. **Splash** → Tap to begin
2. **Upload** → Drag & drop or select an image (JPG/PNG/WEBP)
3. **Mood & Vibe** → Describe the feeling and editing style
4. **Processing** → Watch the AI work its magic ✨
5. **Results** → View before/after, stylized captions, and play music 🎧

---

## 💻 Running the CLI

Prefer the terminal? Run the interactive pipeline directly:

```bash
python Start.py
```

Follow the prompts to specify mood and vibe. Results print to the console.

---

## 📂 Project Structure

```
FrameKraft/
├── backend/
│   ├── app/               # FastAPI web service
│   │   ├── main.py        # Entry point & static file serving
│   │   ├── api/           # API endpoints (/api/pipeline)
│   │   ├── core/          # Reusable pipeline logic
│   │   └── utils/         # Image helpers
│   ├── Start.py           # CLI pipeline entry point
│   ├── Blip.py            # BLIP model wrapper
│   ├── Clip.py            # CLIP model wrapper
│   ├── CvEdit.py          # OpenCV editing engine
│   ├── MusicRag.py        # Music recommendation engine
│   └── requirements.txt   # Python dependencies
└── frontend/
    └── framekraft.html    # Single-page web app
```

---

## 🔌 API Reference

### `POST /api/pipeline`

Upload an image and get back the full pipeline results.

**Request:** `multipart/form-data`
| Field | Type | Description |
|-------|------|-------------|
| `image` | File | JPG/PNG/WEBP image |
| `mood`  | String | e.g., `"melancholic"`, `"joyful"` |
| `vibe`  | String | e.g., `"cinematic warm grain"` |

**Response:** JSON containing `original_image`, `edited_image`, `captions`, `edit_commands`, and `music` recommendations.

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `Address already in use` | Stop existing server or use `--port 8080` |
| `ImportError` | Ensure venv is activated & `pip install` completed |
| Groq API errors | Check `GROQ_API_KEY` in `.env` |
| Audio previews not playing | Requires internet connection for iTunes URLs |

---

## 🤝 Contributing

Feel free to fork, open issues, or submit PRs! When adding features:
- New AI models → create a dedicated module following `Blip.py`/`Clip.py` patterns
- New edit commands → add to `EDIT_LIBRARY` in `CvEdit.py`
- API endpoints → add to `backend/app/api/`
- Frontend → edit `frontend/framekraft.html`

---

**Built with ❤️ and AI. Happy creating!** ✨
