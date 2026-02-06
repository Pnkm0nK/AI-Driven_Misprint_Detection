# AI-Driven label misprint detection
Repository for code during research and development of my bachelor thesis.

This repository contains multiple utility classes to process labels including:
- Image processing
- OCR via Tesseract OCR
- Visualization
- Storing, creating and processing ROIs
- Metric calculation, etc.

To process printed label information, scan pdfs are converted to images. 

 

Setup notes:
- Install Tesseract OCR (recommend the UB Mannheim build). Ensure you know the full path to `tesseract.exe`, for example `C:\Program Files\Tesseract-OCR\tesseract.exe`.
- Install Poppler for Windows and note the folder that contains `pdftoppm.exe` (e.g. `C:\poppler-23.05.0\Library\bin`).
- To enable position independent and multi-symbol DataMatrix detection, the library needs to be compiled with a c++20 compiler. Link: https://pypi.org/project/zxing-cpp/
- Create a .env file to include:
```
POPPLER_PATH = "PATH_TO_POPPLER"
TESSERACT_PATH = "PATH_TO_TESSERACT"
PADDING = 20
```

Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

