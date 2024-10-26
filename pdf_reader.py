import fitz  # PyMuPDF

# Function to read PDF and extract text
def read_pdf(pdf_path):
    try:
        text = ""
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""
