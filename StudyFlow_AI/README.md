# StudyFlow AI

**StudyFlow AI** is a Streamlit multi-agent prototype that turns one academic PDF into structured notes, simple explanations, practice questions, flashcards, and a revision checklist.

## What it does

1. Reads the first 10 pages of one PDF in memory.
2. Extracts selectable text with PyMuPDF.
3. Uses OCR as a fallback for scanned pages when Tesseract is installed.
4. Masks common emails, phone numbers, student-ID labels, and long numeric identifiers before text is sent to the model.
5. Runs five focused agents:
   - Topic Mapping Agent
   - Structured Notes Agent
   - Concept Explanation Agent
   - Practice Agent
   - Quality Check Agent
6. Lets the student download a privacy-safer JSON study pack. The export does not include raw PDF bytes or raw extracted text.

## Important boundaries

- Use course material that you are permitted to process.
- Do not upload confidential records, official transcripts, or sensitive university documents.
- The app produces study support, not guaranteed exam questions.
- It does not replace the source PDF or your instructor's guidance.

## Setup on Windows PowerShell

### 1. Open the project folder

```powershell
cd "C:\Users\leena\OneDrive\Desktop\training tasks\StudyFlow_AI"
```

### 2. Create and activate a virtual environment

```powershell
py -m venv .venv
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& ".\.venv\Scripts\Activate.ps1")
```

### 3. Install the libraries

```powershell
python -m pip install -r requirements.txt
```

### 4. Add your DeepSeek key

Open the `.env` file in VS Code. Change only this line:

```env
DEEPSEEK_API_KEY=
```

Paste your key after `=`. Keep it private. Do not place the key in `app.py`.

The default model is:

```env
LLM_MODEL=deepseek-v4-flash
```

### 5. Run the app

```powershell
streamlit run app.py
```

## Optional: OCR for scanned PDFs

OCR is only needed when a PDF is a scan or mostly an image.

1. Install Tesseract OCR on your computer.
2. If Windows cannot find it automatically, add this line to `.env`:

```env
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

The app can still read ordinary text-based PDFs without Tesseract.

## Test file

A small fake academic PDF is included here:

```text
demo_files/demo_machine_learning_lecture.pdf
```

Upload it to test the full workflow after adding your DeepSeek key.

## Project structure

```text
StudyFlow_AI/
├── app.py                 # Streamlit user interface
├── agents.py              # Five DeepSeek agents
├── pdf_processing.py      # PDF text extraction and optional OCR
├── utils.py               # Redaction and JSON helpers
├── requirements.txt       # Libraries to install
├── .env                   # Add your API key here
├── .env.example           # Safe copy of the environment template
├── .gitignore             # Prevents accidental key uploads
├── .streamlit/config.toml # 10 MB Streamlit upload limit
└── demo_files/
    └── demo_machine_learning_lecture.pdf
```
