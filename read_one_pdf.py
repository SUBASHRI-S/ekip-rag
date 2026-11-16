import fitz  # the PyMuPDF library

doc = fitz.open("data/raw/rbi_nbfc_credit_facilities.pdf")
print("Number of pages:", len(doc))
print(doc[0].get_text()[:500])