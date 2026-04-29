import torch
import nltk
import os
import re

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
def simple_sentence_split(text):
    return text.split(".")

from PyPDF2 import PdfReader
from docx import Document

nltk.download("punkt")

# =========================
# Load Model
# =========================

MODEL_NAME = "t5-small"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

print("Model loaded successfully on", device)

# =========================
# File Readers
# =========================

def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"

    return text


def read_docx(file_path):
    doc = Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])


def read_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        return read_txt(file_path)
    elif ext == ".pdf":
        return read_pdf(file_path)
    elif ext == ".docx":
        return read_docx(file_path)
    else:
        raise ValueError("Unsupported file type. Use .txt, .pdf, or .docx")


# =========================
# Clean Text
# =========================

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


# =========================
# Split Text
# =========================

def split_text(text, max_tokens=350):
    text = clean_text(text)
    sentences = simple_sentence_split(text)

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(tokenizer.encode(sentence, add_special_tokens=False))

        if current_length + sentence_length > max_tokens:
            if current_chunk:
                chunks.append(" ".join(current_chunk))

            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# =========================
# Summarize Chunk
# =========================

def summarize_chunk(chunk, max_length=100, min_length=30):
    chunk = clean_text(chunk)

    input_text = "summarize: " + chunk

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        summary_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=3
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# =========================
# Main Pipeline
# =========================

def summarize_file(file_path):
    text = read_file(file_path)
    text = clean_text(text)

    if len(text) < 50:
        raise ValueError("File text is too short to summarize.")

    print("Text length:", len(text))

    chunks = split_text(text)
    print("Chunks:", len(chunks))

    summaries = []

    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i + 1}/{len(chunks)}")
        summary = summarize_chunk(chunk)
        summaries.append(summary)

    combined_summary = " ".join(summaries)

    if len(chunks) > 1:
        print("Creating final summary...")
        final_summary = summarize_chunk(
            combined_summary,
            max_length=120,
            min_length=40
        )
    else:
        final_summary = combined_summary

    return final_summary


# =========================
# Run Script
# =========================

if __name__ == "__main__":
    file_path = input("Enter file path: ").strip()

    summary = summarize_file(file_path)

    os.makedirs("output", exist_ok=True)

    output_path = "output/summary.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary)

    print("\nFinal Summary:\n")
    print(summary)

    print(f"\n✅ Summary saved to {output_path}")