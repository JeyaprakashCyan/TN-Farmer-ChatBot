import os
import json
import argparse
import pytesseract
from PIL import Image
from typing import List, Dict, Any


def configure_tesseract() -> None:
    if not pytesseract.pytesseract.tesseract_cmd or os.path.exists(pytesseract.pytesseract.tesseract_cmd):
        return

    common_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(common_path):
        pytesseract.pytesseract.tesseract_cmd = common_path
        return

    raise RuntimeError(
        "Tesseract OCR was not found. Install it from https://github.com/tesseract-ocr/tesseract "
        "and add tesseract.exe to PATH."
    )

class SchemeDataProcessor:
    def __init__(self, data_json_path: str):
        with open(data_json_path, "r", encoding="utf-8") as f:
            self.schemes = json.load(f)

    def process_and_chunk(self, chunk_size: int = 400) -> List[Dict[str, Any]]:
        all_chunks = []
        
        for scheme in self.schemes:
            combined_text = f"Scheme Name: {scheme['title']}\nSource URL: {scheme['url']}\n\n"
            combined_text += scheme['raw_text']
            
            # Extract OCR text from accompanying scheme images
            for img_path in scheme.get("image_paths", []):
                if os.path.exists(img_path):
                    try:
                        ocr_content = pytesseract.image_to_string(Image.open(img_path))
                        if len(ocr_content.strip()) > 30:
                            combined_text += f"\n\n[OCR Image Text]\n{ocr_content.strip()}"
                    except Exception as e:
                        print(f"OCR failed for {img_path}: {e}")
                        
            # Chunking logic
            words = combined_text.split()
            for i in range(0, len(words), chunk_size):
                chunk_text = " ".join(words[i:i+chunk_size])
                all_chunks.append({
                    "chunk_id": f"{scheme['scheme_id']}_c{i//chunk_size}",
                    "scheme_title": scheme['title'],
                    "url": scheme['url'],
                    "text": chunk_text,
                    "department": "Agriculture & Farmers Welfare Department",
                    "state": "Tamil Nadu"
                })
                
        return all_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and chunk scheme text with OCR support.")
    parser.add_argument(
        "input",
        nargs="?",
        default="raw_scheme_data/scraped_schemes.json",
        help="Path to the scraper JSON output.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="raw_scheme_data/extracted_chunks.json",
        help="Path for the extracted chunks JSON output.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=400,
        help="Maximum number of words per chunk.",
    )
    args = parser.parse_args()

    if args.chunk_size <= 0:
        parser.error("--chunk-size must be greater than zero")
    if not os.path.isfile(args.input):
        parser.error(f"Input file not found: {args.input}")

    configure_tesseract()
    chunks = SchemeDataProcessor(args.input).process_and_chunk(args.chunk_size)
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(chunks, output_file, indent=2, ensure_ascii=False)
    print(f"Extracted {len(chunks)} chunks to {args.output}")


if __name__ == "__main__":
    main()