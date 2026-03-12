from pathlib import Path
import cv2
import numpy as np
import random
import string
import time
import re
from modules.ImageProcessor import ImageProcessor

import pyperclip
import pyautogui
import pytesseract
from PIL import ImageGrab

PAUSE = 0.5          # seconds between actions for pyautogui
DRAG_DURATION = 0.5 
SAVE_DIR = Path("./parsed_data")       # directory to save screenshots

def check_already_exists(code: str) -> bool:
    """Check if a file with the given code already exists in the SAVE_DIR."""
    for subdir in ["151", "146", "107"]:
        if (SAVE_DIR / subdir / f"{code}.png").exists():
            print(f"File with code {code} already exists in {subdir}. Skipping.")
            return True
    return False

def random_unique_code() -> str:
    """Generate a unique code: 2 non-zero digits + one of SF/SG/SL."""
    batch_letters = ["SF", "SG", "SL", "TA", "TB", "TC", "TD", "TE", "TF", "TG", "TH", "TJ", "TK", "TL", "TM", "TN", "TP"]

    def _generate() -> str:
        digits = "".join(random.choices(string.digits, k=2))
        while digits == "00":
            digits = "".join(random.choices(string.digits, k=2))
        return digits + random.choice(batch_letters)

    code = _generate()
    while check_already_exists(code):
        code = _generate()
    return code


def sanitize_filename(text: str) -> str:
    """Strip characters that are not valid in filenames."""
    text = text.strip()
    text = text[:5]
    text = re.sub(r'[\\/*?:"<>|\r\n]', "_", text)
    return text 


def ocr_region(x1: int, y1: int, x2: int, y2: int) -> str:
    """Capture a screen region and return OCR text."""
    screenshot = ImageGrab.grab(bbox=(x1, y1, x2, y2))
    config = "--oem 3 --psm 6 -c textord_disable_skew=1"
    return pytesseract.image_to_string(screenshot, config=config).strip()


def get_image_from_clipboard() -> np.ndarray | None:
    """Return the clipboard image as a BGR numpy array, or None if no image is present."""
    img = ImageGrab.grabclipboard()
    if img is None:
        print("[clipboard] No image found in clipboard.")
        return None
    img_np = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    print(f"[clipboard] Image grabbed: {img_np.shape}")
    return img_np


def get_text_via_clipboard(x: int, y: int) -> str:
    """Click a cell, select all its text, copy, and return the clipboard content."""
    pyperclip.copy("")  # clear clipboard first
    pyautogui.click(x, y)
    pyautogui.click(x, y, clicks=3, interval=0.1)
    time.sleep(0.1)   
    pyautogui.rightClick(x, y)
    pyautogui.click(x + 30, y + 20)  # Click on the "Copy" option in the context menu
    result = pyperclip.paste().strip()
    print(f"[clipboard raw]: '{result}'")
    return result


def main():
    print("Starting in 5 seconds — switch to your browser window now...")
    time.sleep(5)

    pyautogui.PAUSE = PAUSE
    pyautogui.FAILSAFE = True  # move mouse to top-left corner to abort

    for _ in range(30):
        # --- Step 1: Click reprint ---
        code = random_unique_code()
        print(f"Generated code: {code}")
        pyautogui.click(450, 500, duration=0.2)  # click with a slight duration for reliability
        time.sleep(2)
        pyautogui.typewrite(code, interval=0.05)
        pyautogui.press("enter")
        print("Typed code and pressed Enter.")

        # --- Step 2: Click Search ---
        pyautogui.click(450, 420)
        print("Clicked search.")
        time.sleep(5)

        # --- Step 3: OCR region (115,540)-(150,555) to get label text ---
        coords_ocr = (121, 545, 200, 570)
        ocr_text = ocr_region(*coords_ocr)
        print(f"OCR result: '{ocr_text}'")
        while len(ocr_text) < 4:
            print("No labels found. Retrying search...")
            pyautogui.click(160, 365)
            pyautogui.hotkey("ctrl", "a")
            pyautogui.press("backspace")
            code = random_unique_code()
            pyautogui.typewrite(code, interval=0.05)
            pyautogui.press("enter")
            print("Typed code and pressed Enter.")
            pyautogui.click(450, 420)
            print("Clicked search.")
            time.sleep(5)
            ocr_text = ocr_region(*coords_ocr)

        clipboard_text = get_text_via_clipboard(137, 550)
        filename_base = sanitize_filename(clipboard_text)

        # --- Step 4: Click Row---
        pyautogui.click(140, 555)
        print("Clicked row.")

        pyautogui.click(320,850, duration=0.2)
        print("Clicked continue.")
        time.sleep(4.5)

        pyautogui.click(560,645, duration=0.2)
        print("Clicked print preview.")
        time.sleep(18)

        # --- Step 5: Drag (320,320) -> (50,50) ---
        pyautogui.moveTo(880, 530, duration=0.2)
        pyautogui.dragTo(100, 320, duration=DRAG_DURATION, button="left")
        print("Dragged preview.")

        # --- Step 6: Drag template matched results to (1900,990) ---
        processor = ImageProcessor()
        # Capture full screen and convert to numpy array for template matching
        screen = ImageGrab.grab()
        screen_np = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)
        _, loc = processor.get_template_matching_results(screen_np, "./drag.png")
        x, y = loc
        pyautogui.moveTo(x, y, duration=0.2)
        pyautogui.dragTo(1930, 1000, duration=DRAG_DURATION, button="left")
        print("Dragged to improve quality")

        screen = ImageGrab.grab()
        screen_np = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)
        _, loc = processor.get_template_matching_results(screen_np, "./fit.png")
        x, y = loc
        pyautogui.click(x+40, y+10, duration=0.2)
        print("fitted to window")
        pyautogui.rightClick(900,540, duration=0.2)
        pyautogui.click(910,600, duration=0.2)
        print("Clicked 'Copy image' from context menu.")
        screen = ImageGrab.grab()
        screen_np = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)
        _, loc = processor.get_template_matching_results(screen_np, "./close.png")
        x, y = loc
        pyautogui.click(x+30, y+13, duration=0.2)
        print("closed")
        pyautogui.click(50,260, duration=0.2)

        # --- Get copied image from clipboard ---
        clipboard_img = get_image_from_clipboard()
        if clipboard_img is not None:
            if "151" in filename_base:
                save_path = SAVE_DIR/ "151" / f"{code}.png"
            elif "146" in filename_base:
                save_path = SAVE_DIR/ "146" / f"{code}.png"
            elif "107" in filename_base:
                save_path = SAVE_DIR/ "107" / f"{code}.png"
            else:
                save_path = SAVE_DIR / f"{code}.png"
            cv2.imwrite(str(save_path), clipboard_img)
            print(f"Clipboard image saved to: {save_path}")
    
        time.sleep(5)



if __name__ == "__main__":
    main()
