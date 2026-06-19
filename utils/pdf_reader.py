from pypdf import PdfReader


def extract_pdf_text(uploaded_file, max_chars: int = 50000) -> str:
    reader = PdfReader(uploaded_file)

    if not reader.pages:
        raise ValueError("The uploaded PDF has no pages.")

    pages_text = []
    for page in reader.pages:
        try:
            extracted = page.extract_text()
            if extracted:
                pages_text.append(extracted)
        except Exception:
            pass

    full_text = "\n".join(pages_text).strip()

    if not full_text:
        raise ValueError(
            "Could not extract any text from this PDF. "
            "It may be a scanned image PDF. Please try a text-based PDF."
        )

    return full_text[:max_chars]
