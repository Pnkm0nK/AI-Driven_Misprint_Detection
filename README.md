# Bachelor_thesis_OCR
Repository for code during research and development of my bachelor thesis
 
Using Tesseract OCR on PDF

This repository includes a small script `tesseract_test.py` that converts PDF pages to images and runs Tesseract OCR to extract text.

Windows setup notes:
- Install Tesseract OCR (recommend the UB Mannheim build). Ensure you know the full path to `tesseract.exe`, for example `C:\Program Files\Tesseract-OCR\tesseract.exe`.
- Install Poppler for Windows and note the folder that contains `pdftoppm.exe` (e.g. `C:\poppler-23.05.0\Library\bin`).

Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the script (PowerShell example):

```powershell
# Basic usage (specify poppler and tesseract if not on PATH)
python tesseract_test.py "input.pdf" --poppler-path "C:\\poppler-xxx\\Library\\bin" \
	--tesseract-cmd "C:\\Program Files\\Tesseract-OCR\\tesseract.exe" --output output.txt

# If Tesseract and Poppler are on PATH, a simpler command works:
python tesseract_test.py "input.pdf" --output output.txt
```

The script writes extracted text to `output.txt` (or `input.txt` if you omit `--output`).

If you run into issues, check that the `pdftoppm.exe` and `tesseract.exe` paths are correct and that the language data you request with `--lang` is installed for Tesseract.
