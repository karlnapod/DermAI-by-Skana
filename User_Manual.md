# DermAI by Skana — User Manual

**Version:** 1.0  
**Last Updated:** March 2026

---

## Before You Begin

This manual walks you through everything needed to set up and run DermAI by Skana from scratch on your own machine. No prior files will be provided — you will set up your own environment, obtain your own API keys, and configure the application yourself.

By the end of this guide you will have:

- A working Python backend running on port **8000**
- A working React frontend running on port **3000**
- The full application accessible at **http://localhost:3000**

---

## System Requirements

Make sure your machine meets the following minimum requirements before starting.

| Requirement | Minimum |
|---|---|
| Operating System | Windows 10/11, macOS 12+, or Ubuntu 20.04+ |
| RAM | 8 GB (16 GB recommended for smooth model loading) |
| Disk Space | 2 GB free (plus space for model files) |
| Python | 3.12.x (via conda — see Step 1) |
| Node.js | 18.x or later |
| npm | 9.x or later (included with Node.js) |
| Internet | Required for API key setup and npm/pip downloads |

---

## What You Will Need

Before starting, gather the following:

1. **The DermAI codebase** — the folder you are reading this from.
2. **Two trained model weight files:**
   - `best_model.pth` — the skin cancer detection model
   - `best_model.pt` — the skin disease classification model
3. **Google AI Studio API keys** — free to obtain at [aistudio.google.com](https://aistudio.google.com). You will need up to **three keys** (one for the disease description generator, one backup key for the same, and one for the chatbot). You can use the same key for all three if you only have one, but separate keys are recommended to avoid hitting rate limits.

---

## Part 1 — Python Backend Setup

### Step 1 — Install Conda and Create the Python Environment

DermAI requires Python **3.12**. The easiest way to manage this is with Conda.

**1a. Download Miniconda**

Go to [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html) and download the installer for your operating system. Choose the **Python 3.12** installer or any recent version — Conda can create environments with specific Python versions regardless.

Run the installer and follow the on-screen instructions.

**1b. Open a Terminal**

- **Windows:** Open "Anaconda Prompt" or "Miniconda Prompt" from the Start Menu.
- **macOS / Linux:** Open your regular Terminal application.

**1c. Create the DermAI Environment**

```bash
conda create -n dermai python=3.12.7
```

When prompted `Proceed ([y]/n)?`, type `y` and press Enter. This creates an isolated Python 3.12.7 environment named `dermai`.

**1d. Activate the Environment**

```bash
conda activate dermai
```

You should see `(dermai)` appear at the beginning of your terminal prompt. You must keep this environment active any time you run the backend.

---

### Step 2 — Install Python Dependencies

Navigate to the `backend` folder inside the DermAI project:

**Windows:**
```
cd "C:\path\to\DermAI by Skana\backend"
```

**macOS / Linux:**
```bash
cd "/path/to/DermAI by Skana/backend"
```

> Replace the path with the actual location of the DermAI folder on your machine.

Now install all required packages:

```bash
pip install -r requirements.txt
```

This will install FastAPI, PyTorch (CPU build), Pillow, the Google AI Studio SDK, and all other dependencies. The download may take a few minutes as PyTorch is a large package.

**If you see an error that says "no matching distribution found" for torch:**

Make sure your `requirements.txt` still contains the following line at the top (it should already be there):
```
--extra-index-url https://download.pytorch.org/whl/cpu
```
This line tells pip to look for PyTorch at the official PyTorch wheel server, not just PyPI.

---

### Step 3 — Place the Model Weight Files

The AI models are not included in the codebase because of their large file sizes. You need to place them in the correct folder manually.

**3a. Create the models folder** (if it does not already exist):

**Windows:**
```
mkdir "C:\path\to\DermAI by Skana\backend\models"
```

**macOS / Linux:**
```bash
mkdir -p "/path/to/DermAI by Skana/backend/models"
```

**3b. Place the model files:**

Copy both model files into the `backend/models/` folder:

| File | Purpose |
|---|---|
| `best_model.pth` | Skin cancer detection (ResNet-50) |
| `best_model.pt` | Skin disease classification (ResNet-152) |

After this step, your `backend/models/` folder should look like:

```
backend/
└── models/
    ├── best_model.pth
    └── best_model.pt
```

---

### Step 4 — Get Your Google AI Studio API Keys

The chatbot and AI description generator require API keys from Google AI Studio.

**4a. Go to Google AI Studio**

Open [https://aistudio.google.com](https://aistudio.google.com) in your browser. Sign in with a Google account.

**4b. Create API Keys**

1. Click **"Get API key"** in the top-right corner (or find it in the left navigation).
2. Click **"Create API key"** and select or create a Google Cloud project.
3. Copy the key that is generated. Treat this like a password — do not share it.

Repeat this process to create up to **three keys** (one primary, one backup for descriptions, one for the chatbot). If you only create one key, you can use the same key for all three fields — the application will still work, but you are more likely to hit rate limits.

---

### Step 5 — Configure the Environment Variables

The application reads API keys from a file called `.env` in the `backend` folder.

**5a. Open the `.env` file**

Navigate to `backend/.env` in your file explorer or text editor. The file already exists and looks like this:

```
GEMINI_PRIMARY_API_KEY=
GEMINI_BACKUP_API_KEY=
GEMMA_API_KEY=
```

**5b. Paste your API keys**

Fill in your keys on the right side of the `=` sign, with no spaces and no quotation marks:

```
GEMINI_PRIMARY_API_KEY=AIzaSy...your_key_here...
GEMINI_BACKUP_API_KEY=AIzaSy...your_backup_key_here...
GEMMA_API_KEY=AIzaSy...your_chatbot_key_here...
```

> If you only have one key, paste the same key for all three fields.

**5c. Save and close the file.**

> **Security note:** Never share your `.env` file. Never paste your API keys into a chat, email, or code comment. The `.env` file is intentionally excluded from version control.

---

### Step 6 — Start the Backend Server

Make sure:
- Your `(dermai)` conda environment is active
- You are in the `backend` folder

Then run:

```bash
uvicorn main:app --reload --port 8000
```

**What you should see:**

```
INFO:     Will watch for changes in these directories: ['.../backend']
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [...]
[CancerClassifier] Loaded. CONFIDENCE_THRESHOLD=0.4, CLASSIFICATION_THRESHOLD=0.5
[DiseaseClassifier] Loaded from .../backend/models/best_model.pt
INFO:     Application startup complete.
```

If you see the `[CancerClassifier] Loaded` and `[DiseaseClassifier] Loaded` messages, both models are ready.

**If a model is not found**, the server will print a warning but still start:
```
Warning: cancer model not found at .../models/best_model.pth.
Place best_model.pth inside the backend/models/ folder.
```
Go back to Step 3 and check the file names and location.

**To verify the backend is running**, open [http://localhost:8000/health](http://localhost:8000/health) in your browser. You should see:
```json
{
  "status": "ok",
  "cancer_model_loaded": true,
  "disease_model_loaded": true
}
```

---

### Stopping the Backend

To stop the server, click on the terminal window where `uvicorn` is running and press **Ctrl+C**.

If Ctrl+C does not respond, open a new terminal and run one of the following:

**Windows:**
```powershell
# Kill by port
$p = (Get-NetTCPConnection -LocalPort 8000 -State Listen).OwningProcess
Stop-Process -Id $p -Force
```

**macOS / Linux:**
```bash
lsof -ti:8000 | xargs kill -9
```

---

## Part 2 — Frontend Setup

### Step 7 — Install Node.js

If you do not already have Node.js installed:

1. Go to [https://nodejs.org](https://nodejs.org)
2. Download the **LTS (Long Term Support)** version for your operating system
3. Run the installer and follow the on-screen instructions

Verify the installation by opening a new terminal and running:
```bash
node --version
npm --version
```
Both commands should print a version number (e.g., `v20.x.x` and `10.x.x`).

---

### Step 8 — Install Frontend Dependencies

Open a **new terminal** (separate from the one running the backend). Navigate to the `frontend` folder:

**Windows:**
```
cd "C:\path\to\DermAI by Skana\frontend"
```

**macOS / Linux:**
```bash
cd "/path/to/DermAI by Skana/frontend"
```

Install all required Node packages:

```bash
npm install
```

This reads the `package.json` file and downloads all frontend dependencies into a `node_modules` folder. This may take 1–2 minutes the first time.

---

### Step 9 — Start the Frontend Development Server

From the `frontend` folder, run:

```bash
npm run dev
```

**What you should see:**

```
  VITE v6.x.x  ready in 300ms

  ➜  Local:   http://localhost:3000/
  ➜  Network: use --host to expose
```

Open [http://localhost:3000](http://localhost:3000) in your browser. The DermAI application should now be visible.

---

## Part 3 — Using the Application

### Overview of Pages

| Page | How to Access | What It Does |
|---|---|---|
| Home | Click the DermAI logo or "Home" | Landing page with application introduction |
| Disclaimers | Click "Disclaimers" in the nav bar | Medical and AI limitation notices |
| Skin Cancer Detector | Click "Skin Cancer" in the nav bar | Binary cancer image classification |
| Skin Disease Detector | Click "Skin Disease" in the nav bar | 8-class disease image classification |
| About | Click "About" in the nav bar | Project and team information |
| Karl (Chatbot) | Click the teal chat bubble button (bottom-right) | AI dermatology assistant |

---

### Using the Skin Cancer Detector

1. Navigate to the **Skin Cancer** page.
2. Either **drag and drop** a JPG or PNG image onto the upload zone, or **click the zone** to open your file picker and select an image.
3. Once the image appears in the preview, click **"Analyse Image"**.
4. Wait a moment for the model to process the image (typically 1–3 seconds).
5. The results will appear below the image:
   - A **confidence ring** shows how certain the model is (a percentage)
   - One of three result cards will be highlighted:
     - **Green** — the model classifies the image as benign
     - **Red** — the model classifies the image as malignant
     - **Grey** — the model's confidence is too low to make a call (inconclusive)
6. When you are ready to test a new image, click **"Scan A New Image"** to reset the page.

> **What "inconclusive" means:** The model's confidence was below 40%. This most commonly happens when the uploaded image is blurry, low resolution, not a dermoscopic image of skin, or shows a condition the model has not been trained on.

---

### Using the Skin Disease Detector

1. Navigate to the **Skin Disease** page.
2. Upload a JPG or PNG image using drag-and-drop or the file picker.
3. Click **"Analyse Image"**.
4. Results are displayed in three sections:
   - **Class probability bars** — eight bars, one per disease category, filled to their confidence percentage. The longest bar is the model's top prediction.
   - **Main result** — shows the predicted disease name, confidence percentage, and a colour-coded severity bar with a medical urgency label.
   - **AI Description box** — this section shows a "Generate Description" button.
5. Click **"Generate Description"** to request an AI-written explanation of the result. This calls the Google AI Studio API, so it requires an internet connection and a valid API key. It takes approximately 3–8 seconds.
   - A counter in the top-right of the AI Description box shows how many descriptions you have left in the current 2-minute window (maximum 5 per 2 minutes).
6. Click **"Scan A New Image"** to fully reset the page, clear the result, and clear the AI description.

---

### Using Karl — The AI Chatbot

1. Click the **teal speech bubble button** in the bottom-right corner of any page.
2. The chatbot panel slides in.
3. Type your dermatology question in the text box at the bottom.
4. Press **Enter** to send, or click the **send button** (arrow icon).
5. Karl will respond in the chat area. User messages appear on the right; Karl's responses appear on the left.

**Input rules and controls:**

| Feature | How to use it |
|---|---|
| Send message | Press **Enter** |
| New line (don't send) | Press **Shift + Enter** |
| Undo last word | Press **Ctrl + Z** (Windows/Linux) or **Cmd + Z** (Mac) — up to 7 words |
| Undo entire paste | After pasting, press **Ctrl/Cmd + Z** once |
| Copy text | **Ctrl + C** (Windows/Linux) or **Cmd + C** (Mac) |
| Paste text | **Ctrl + V** (Windows/Linux) or **Cmd + V** (Mac) |

**Limits:**

| Limit | Value |
|---|---|
| Maximum characters per message | 500 (including spaces) |
| Maximum messages per session | 15 user messages |

When you reach 15 messages, the input area is replaced with a notice. Refresh the page to start a new conversation.

Karl is strictly scoped to **dermatology topics only**. Asking about anything outside skin health will result in: *"Sorry, I cannot help you with that."*

---

## Part 4 — Troubleshooting

### The frontend shows a blank page or "Cannot GET /"

Make sure:
- The backend is running on port 8000 (`uvicorn main:app --reload --port 8000`)
- The frontend is running on port 3000 (`npm run dev`)
- You are visiting `http://localhost:3000` (not `localhost:8000`)

### "Model is not loaded" error on the detector pages

This means the server started but could not find the model file. Check:
- Both files are in `backend/models/`
- The file names are exactly `best_model.pth` (cancer) and `best_model.pt` (disease)
- The files are not inside a sub-folder inside `models/`

Restart the backend after placing the files.

### "GEMINI_PRIMARY_API_KEY is not set" when generating a description

The `.env` file is missing or the key was not saved correctly. Check:
- The `.env` file exists at `backend/.env` (not `backend/.env.txt`)
- There are no spaces around the `=` sign
- The key was saved and the backend was restarted after editing `.env`

### The conda environment is not recognised in a new terminal

You need to activate the environment each time you open a new terminal:
```bash
conda activate dermai
```

### "pip is not recognised" on Windows

Make sure you opened "Anaconda Prompt" or "Miniconda Prompt" (not the regular Command Prompt or PowerShell) and that your conda environment is activated.

### npm install or npm run dev fails

Try:
1. Make sure you are in the `frontend` folder (the one containing `package.json`)
2. Delete the `node_modules` folder and `package-lock.json`, then re-run `npm install`

### The AI Description box shows an error

This typically means:
- Your API key is invalid or has expired — check `backend/.env` and regenerate a key at aistudio.google.com
- Both the primary and backup keys have hit their rate limits — wait a few minutes and try again
- No internet connection is available

### Killing the backend when Ctrl+C does not work

**Windows:**
```powershell
$p = (Get-NetTCPConnection -LocalPort 8000 -State Listen).OwningProcess
Stop-Process -Id $p -Force
```

**macOS / Linux:**
```bash
lsof -ti:8000 | xargs kill -9
```

---

## Part 5 — Quick Reference

### Starting the Application (Every Time)

**Terminal 1 — Backend:**
```bash
conda activate dermai
cd "path/to/DermAI by Skana/backend"
uvicorn main:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd "path/to/DermAI by Skana/frontend"
npm run dev
```

**Open browser:** [http://localhost:3000](http://localhost:3000)

---

### File Checklist

Before running, verify these files are in place:

| File | Required | Location |
|---|---|---|
| `best_model.pth` | Yes | `backend/models/best_model.pth` |
| `best_model.pt` | Yes | `backend/models/best_model.pt` |
| `.env` (with keys filled) | Yes | `backend/.env` |
| `node_modules/` folder | Yes (after `npm install`) | `frontend/node_modules/` |

---

### API Key Summary

| Key Name in .env | Used For | Where to Get |
|---|---|---|
| `GEMINI_PRIMARY_API_KEY` | Disease description generation (primary) | aistudio.google.com |
| `GEMINI_BACKUP_API_KEY` | Disease description generation (automatic fallback on rate limit) | aistudio.google.com |
| `GEMMA_API_KEY` | Karl the chatbot | aistudio.google.com |

---

## Medical Disclaimer

DermAI by Skana is an educational and research tool. The AI models provide classifications based on patterns learned from training data and may not be accurate for all cases. **Results from DermAI should never be used as a substitute for professional medical advice, diagnosis, or treatment.** If you have concerns about a skin condition, consult a qualified dermatologist or medical professional.
