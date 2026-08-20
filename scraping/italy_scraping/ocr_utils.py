import os
import time
import platform
import subprocess
import pdfplumber
import docx
import pytesseract
from striprtf.striprtf import rtf_to_text
from pdf2image import convert_from_path

SOURCE_FOLDER = "documents"
OUTPUT_FOLDER = "txts"

def extract_pdf(file_path):
    text_content = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text: text_content.append(text)
    except Exception: pass
                
    extracted = "\n".join(text_content).strip()
    
    if len(extracted) < 50:
        ocr_text = []
        # dpi=150 is fast; thread_count=4 helps the Mac image conversion
        images = convert_from_path(file_path, dpi=150, thread_count=4)
        for image in images:
            text = pytesseract.image_to_string(image, lang='ita')
            ocr_text.append(text)
        return "\n".join(ocr_text), True # Return True to flag it was OCR'd
    return extracted, False

def extract_docx(fp):
    return "\n".join([p.text for p in docx.Document(fp).paragraphs])

def extract_rtf(fp):
    with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
        return rtf_to_text(f.read())

def extract_doc(fp):
    if platform.system() == "Darwin": 
        return subprocess.run(['textutil', '-convert', 'txt', '-stdout', fp], capture_output=True, text=True).stdout
    return ""

def process_single_file(file_path):
    file_name = os.path.basename(file_path)
    input_file_path = os.path.join(SOURCE_FOLDER, file_name)
    base_name, ext = os.path.splitext(file_name)
    output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}.txt")
    
    if os.path.exists(output_path):
        return ("SKIP", file_name, 0, False)

    start_time = time.time()
    was_ocr = False
    try:
        ext = ext.lower()
        if ext == '.pdf':
            text, was_ocr = extract_pdf(input_file_path)
        elif ext == '.docx':
            text = extract_docx(input_file_path)
        elif ext == '.rtf':
            text = extract_rtf(input_file_path)
        elif ext == '.doc':
            text = extract_doc(input_file_path)
        else:
            return ("ERROR", f"Unsupported extension: {ext}", 0, False)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        duration = time.time() - start_time
        return ("SUCCESS", file_name, duration, was_ocr)
    except Exception as e:
        return ("ERROR", f"{file_name}: {str(e)}", 0, False)