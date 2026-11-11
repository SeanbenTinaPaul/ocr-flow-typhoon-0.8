"""
Typhoon OCR 2 (0.8B) & Surya OCR (0.65B) Python Runner
Engine: Local Ollama on GTX 1050 (3GB VRAM)

Adapted from Typhoon 3B patterns with key enhancements for Typhoon OCR 2 0.8B:
1. Single-model execution with session keep-alive to avoid VRAM reloads across PDF pages.
2. Calibrated 16,384 context window so ~2.4k vision tokens leave ~14k for text.
3. Official Infinity-Parser2 prompt template for Typhoon OCR 2 0.8B.
4. Robust JSON parsing for element-level bounding boxes and reading order.
5. Dual output: Clean Markdown (.md) and structured Layout JSON (.json).
6. Multi-page PDF management: page range selection (--pages), incremental saving, and error resilience.
7. Recursive directory scanning (--recursive).
"""

import os
import sys
import glob
import json
import re
import base64
import io
import time
import urllib.request
import urllib.error
import argparse
import pathlib

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_TYPHOON_MODEL = "typhoon-ocr:0.8b"
DEFAULT_SURYA_MODEL = "surya-ocr:0.65b"

# Hardcode your default document/image/folder path here:
# When set, you can simply run: python run_ocr_0.8b.py --model surya
DEFAULT_INPUT_PATH = r""

# Optional: Hardcode your default output folder here:
# Leave empty (r"") to automatically save in the same folder as the input document.
DEFAULT_OUTPUT_DIR = r""

# Official prompt for Typhoon OCR 2 (0.8B)
TYPHOON_08B_PROMPT = """- Extract layout information from the provided PDF image.
- For each layout element, output its bbox, category, and the text content within the bbox.
- Bbox format: [x1, y1, x2, y2].
- Allowed layout categories: ['header', 'title', 'text', 'figure', 'table', 'formula', 'figure_caption', 'table_caption', 'formula_caption', 'figure_footnote', 'table_footnote', 'page_footnote', 'footer'].
- Text extraction and formatting:
  1) For 'figure', the text field must be an empty string.
  2) For 'formula', format text as LaTeX.
  3) For 'table', format text as HTML.
  4) For all other categories (e.g., text, title), format text as Markdown.
- The output text must be exactly the original text from the image, with no translation or rewriting.
- Sort all layout elements in human reading order.
- Final output must be a single JSON object."""

# Official prompt for Surya OCR 2 (0.65B)
SURYA_065B_PROMPT = (
    "OCR this image to HTML. Each block is a div with data-label and data-bbox (x0 y0 x1 y1, normalized 0-1000)."
)


def format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration (e.g., '12.4s' or '2m 15.3s')."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}m {secs:.1f}s"


def parse_page_range(page_spec: str, total_pages: int) -> list:
    """
    Parse page specifications such as '1-3', '5', '1,3,5', '2-', '-4'.
    Returns a sorted, 0-based list of unique valid page indices.
    """
    selected = set()
    parts = [p.strip() for p in page_spec.split(",") if p.strip()]
    for part in parts:
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str.strip()) if start_str.strip() else 1
            end = int(end_str.strip()) if end_str.strip() else total_pages
            start = max(1, min(start, total_pages))
            end = max(1, min(end, total_pages))
            for p in range(start, end + 1):
                selected.add(p - 1)
        else:
            try:
                p = int(part)
                if 1 <= p <= total_pages:
                    selected.add(p - 1)
            except ValueError:
                pass
    res = sorted(list(selected))
    return res if res else list(range(total_pages))


def parse_typhoon_elements(raw: str) -> list:
    """
    Parse Typhoon 0.8B output: a JSON list of layout elements inside a ```json fence.
    Includes graceful recovery if generation was stopped mid-token.
    """
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*(?:```|$)", raw, re.DOTALL)
    if m:
        text = m.group(1).strip()
    else:
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1 and end > start:
            text = raw[start : end + 1]
        else:
            text = raw.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Salvage elements if output cut off at token limit
        cut = text.rfind('{"bbox":')
        if cut > 0:
            salvaged = text[:cut].rstrip().rstrip(",") + "]"
            try:
                return json.loads(salvaged)
            except Exception:
                pass
    return []


def elements_to_markdown(elements: list, keep_header_footer: bool = False, embed_figures: bool = True) -> str:
    """
    Convert Typhoon OCR 2 structured elements into clean Markdown in reading order.
    Preserves visual figure bounding boxes so downstream croppers can extract images.
    """
    skip = set() if keep_header_footer else {"header", "footer", "page_footnote"}
    lines = []
    for elem in elements:
        category = elem.get("category", "")
        text = elem.get("text", "").strip()
        bbox = elem.get("bbox")

        if category in skip:
            continue

        if category == "figure":
            if embed_figures and bbox and len(bbox) == 4:
                bbox_str = " ".join(str(int(x)) for x in bbox)
                lines.append(f'<div data-bbox="{bbox_str}" data-label="Figure"><img/></div>')
            continue

        if not text:
            continue

        if category == "title":
            lines.append(f"# {text}")
        elif category in ("header", "figure_caption", "table_caption"):
            lines.append(f"**{text}**")
        elif category == "table":
            lines.append(f"\n{text}\n")
        elif category == "formula":
            clean_formula = text.strip()
            # Strip existing delimiters to prevent double-wrapping ($$ $$...$$ $$)
            if clean_formula.startswith("$$") and clean_formula.endswith("$$") and len(clean_formula) >= 4:
                clean_formula = clean_formula[2:-2].strip()
            elif clean_formula.startswith("$") and clean_formula.endswith("$") and len(clean_formula) >= 2:
                clean_formula = clean_formula[1:-1].strip()

            # Sanitize common OCR LaTeX errors:
            # 1. \mug -> \mu\text{g}
            clean_formula = re.sub(r"\\mug\b", r"\\mu\\text{g}", clean_formula)
            # 2. \overline inside \text{...} -> move \overline outside \text{}
            clean_formula = re.sub(
                r"\\text\s*\{\s*([^{}]*?)\s*\(\\overline\s*\{\s*([^{}]+)\s*\}\)\s*([^{}]*?)\}",
                r"\\text{\1} (\\overline{\2}) \\text{\3}",
                clean_formula,
            )
            clean_formula = re.sub(
                r"\\text\s*\{\s*([^{}]*?)\s*\\overline\s*\{\s*([^{}]+)\s*\}\s*([^{}]*?)\}",
                r"\\text{\1} \\overline{\2} \\text{\3}",
                clean_formula,
            )
            lines.append(f"\n$$\n{clean_formula}\n$$\n")
        else:
            # Sanitize stray LaTeX symbols in plain text paragraphs
            clean_text = re.sub(r"\\mug\b", r"µg", text)
            lines.append(clean_text)
    return "\n\n".join(lines)


def ocr_image(
    base64_image: str,
    model: str = DEFAULT_TYPHOON_MODEL,
    prompt: str = None,
    num_ctx: int = 16384,
    num_predict: int = 4096,
    keep_alive=0,
) -> str:
    """
    Send base64-encoded image to local Ollama instance.
    Enforces single-model discipline (keep_alive: 0 or timed) and 16k context window.
    """
    if prompt is None:
        prompt = TYPHOON_08B_PROMPT if "typhoon" in model.lower() else SURYA_065B_PROMPT

    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [base64_image],
            }
        ],
        "stream": False,
        "keep_alive": keep_alive,
        "options": {
            "temperature": 0.0,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }

    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result.get("message", {}).get("content", "")
    except urllib.error.URLError as e:
        print(f"[-] Ollama API error: {e}")
        return ""


def process_file(
    file_path: str,
    model: str = DEFAULT_TYPHOON_MODEL,
    scale: float = 2.0,
    output_dir: str = None,
    keep_header_footer: bool = False,
    max_tokens: int = 4096,
    context_size: int = 16384,
    pages: str = None,
    post_crop: bool = False,
):
    path = pathlib.Path(file_path).resolve()
    if not path.exists():
        print(f"[-] File not found: {path}")
        return

    out_base = pathlib.Path(output_dir) if output_dir else path.parent
    out_base.mkdir(parents=True, exist_ok=True)

    ext = path.suffix.lower()

    if ext == ".pdf":
        backend = None
        try:
            import pypdfium2 as pdfium
            backend = "pypdfium2"
        except ImportError:
            try:
                import fitz
                backend = "pymupdf"
            except ImportError:
                try:
                    import pymupdf as fitz
                    backend = "pymupdf"
                except ImportError:
                    print(f"[-] No PDF renderer found. For PDF rendering, install: pip install pypdfium2 (or PyMuPDF)")
                    return

        print(f"\n[+] Processing PDF ({backend}): {path.name}")
        out_md = out_base / f"{path.stem}.md"
        out_json = out_base / f"{path.stem}.json"

        try:
            doc = pdfium.PdfDocument(path) if backend == "pypdfium2" else fitz.open(path)
            total_pages = len(doc)
            page_indices = parse_page_range(pages, total_pages) if pages else list(range(total_pages))
            print(f"  - Total pages in PDF: {total_pages}. Processing {len(page_indices)} page(s): {[p + 1 for p in page_indices]}")

            all_pages_md = []
            all_pages_elements = []
            doc_start = time.time()

            for idx, page_idx in enumerate(page_indices):
                human_page = page_idx + 1
                is_last_page = (idx == len(page_indices) - 1)
                # Keep model in VRAM during multi-page document processing, unload on the last page
                page_keep_alive = 0 if is_last_page else "5m"

                print(f"\n  - Processing Page {human_page}/{total_pages} (rendering at {scale}x)...")
                page_start = time.time()
                try:
                    if backend == "pypdfium2":
                        page = doc[page_idx]
                        bitmap = page.render(scale=scale)
                        pil_img = bitmap.to_pil()
                        buf = io.BytesIO()
                        pil_img.save(buf, format="PNG")
                        b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")
                        bitmap.close()
                    else:
                        page = doc[page_idx]
                        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
                        b64_img = base64.b64encode(pix.tobytes("png")).decode("utf-8")
                        pix = None  # Free pixmap buffer from memory

                    print(f"    Sending to {model} (num_ctx: {context_size}, keep_alive: {page_keep_alive})...")
                    raw_output = ocr_image(
                        b64_img,
                        model=model,
                        num_ctx=context_size,
                        num_predict=max_tokens,
                        keep_alive=page_keep_alive,
                    )

                    if "typhoon" in model.lower():
                        elements = parse_typhoon_elements(raw_output)
                        md = elements_to_markdown(elements, keep_header_footer=keep_header_footer)
                        all_pages_elements.append({"page": human_page, "elements": elements})
                        all_pages_md.append(f"## Page {human_page}\n\n{md if md else raw_output}")
                    else:
                        all_pages_md.append(f"## Page {human_page}\n\n{raw_output}")

                    # Incremental write so progress is preserved if interrupted
                    with open(out_md, "w", encoding="utf-8") as f:
                        f.write("\n\n---\n\n".join(all_pages_md))
                    if all_pages_elements:
                        with open(out_json, "w", encoding="utf-8") as f:
                            json.dump(all_pages_elements, f, ensure_ascii=False, indent=2)

                    page_elapsed = time.time() - page_start
                    print(f"    Page {human_page} completed in {format_duration(page_elapsed)}")

                except Exception as page_err:
                        print(f"  [-] Error on page {human_page}: {page_err}")
                        all_pages_md.append(f"## Page {human_page}\n\n> [!WARNING]\n> OCR failed on page {human_page}: {page_err}")
                        with open(out_md, "w", encoding="utf-8") as f:
                            f.write("\n\n---\n\n".join(all_pages_md))

            doc_elapsed = time.time() - doc_start
            avg_str = f", avg {format_duration(doc_elapsed / len(page_indices))}/page" if page_indices else ""
            print(f"\n  [=>] Saved Markdown to: {out_md}")
            if all_pages_elements:
                print(f"  [=>] Saved Layout JSON to: {out_json}")
            print(f"  [✓] Total OCR time for {path.name}: {format_duration(doc_elapsed)} ({len(page_indices)} page(s){avg_str})")

        except Exception as e:
            print(f"[-] Error processing PDF {path.name}: {e}")
        finally:
            if "doc" in locals() and hasattr(doc, "close"):
                try:
                    doc.close()
                except Exception:
                    pass

    elif ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"):
        print(f"\n[+] Processing Image: {path.name}")
        img_start = time.time()
        with open(path, "rb") as f:
            b64_img = base64.b64encode(f.read()).decode("utf-8")

        print(f"  - Sending to {model} (num_ctx: {context_size}, keep_alive: 0)...")
        raw_output = ocr_image(
            b64_img,
            model=model,
            num_ctx=context_size,
            num_predict=max_tokens,
            keep_alive=0,
        )

        if "typhoon" in model.lower():
            elements = parse_typhoon_elements(raw_output)
            md = elements_to_markdown(elements, keep_header_footer=keep_header_footer)

            out_md = out_base / f"{path.stem}.md"
            with open(out_md, "w", encoding="utf-8") as f:
                f.write(md if md else raw_output)
            print(f"  [=>] Saved Markdown to: {out_md}")

            if elements:
                out_json = out_base / f"{path.stem}.json"
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(elements, f, ensure_ascii=False, indent=2)
                print(f"  [=>] Saved Layout JSON to: {out_json}")
        else:
            out_html = out_base / f"{path.stem}.html"
            with open(out_html, "w", encoding="utf-8") as f:
                f.write(raw_output)
            print(f"  [=>] Saved HTML to: {out_html}")

        img_elapsed = time.time() - img_start
        print(f"  [✓] Total OCR time for {path.name}: {format_duration(img_elapsed)}")
    else:
        print(f"[-] Unsupported file type: {ext}")
        return

    if post_crop:
        target_ocr_file = out_md if ("out_md" in locals() and out_md.exists()) else (out_html if ("out_html" in locals() and out_html.exists()) else None)
        if target_ocr_file and target_ocr_file.exists():
            try:
                from surya_post_ocr import process_surya_post_ocr
                print(f"\n[+] Triggering Automatic Post-OCR Cropper & Integrator...")
                process_surya_post_ocr(
                    ocr_file=target_ocr_file,
                    source_file=path,
                    scale=scale,
                    page_filter=pages,
                    auto_caption=True,
                )
            except Exception as crop_err:
                print(f"[-] Automatic post-crop error: {crop_err}")


def main():
    parser = argparse.ArgumentParser(
        description="Run local small OCR models (Typhoon OCR 2 0.8B / Surya 0.65B) via Ollama."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Path to an image file, PDF file, or directory containing documents.",
    )
    parser.add_argument(
        "--model",
        choices=["typhoon", "surya"],
        default="typhoon",
        help="OCR model to use (default: typhoon)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="Rendering scale for PDFs (default: 2.0x = ~144 DPI)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Directory to save output files (defaults to same folder as input)",
    )
    parser.add_argument(
        "--keep-header-footer",
        action="store_true",
        help="Include header/footer elements in rendered Markdown",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max output tokens to generate (default: 4096)",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=16384,
        help="Context window size (default: 16384)",
    )
    parser.add_argument(
        "--pages",
        "-p",
        type=str,
        default=None,
        help="Page range to process for PDFs (e.g. '1-3', '5', '1,3,5', '2-'). Defaults to all pages.",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Scan directory recursively for documents (matches 3B example pattern)",
    )
    parser.add_argument(
        "--post-crop",
        action="store_true",
        help="Automatically run post-OCR cropping & integration (<figure>/<figcaption> with micro-OCR) after OCR finishes",
    )

    args = parser.parse_args()

    model_tag = DEFAULT_TYPHOON_MODEL if args.model == "typhoon" else DEFAULT_SURYA_MODEL

    target_input = args.input or DEFAULT_INPUT_PATH

    if not target_input:
        print("Usage: python run_ocr_0.8b.py [path_to_image_or_pdf] [--model typhoon|surya] [--scale 2.0] [--pages 1-3] [--recursive]")
        print("Example: python run_ocr_0.8b.py document.pdf --model typhoon --pages 1-5")
        print("Tip: You can also set DEFAULT_INPUT_PATH at line 32 in run_ocr_0.8b.py to omit the path argument.")
        return

    target_output_dir = args.output_dir or (DEFAULT_OUTPUT_DIR if DEFAULT_OUTPUT_DIR.strip() else None)
    input_path = pathlib.Path(target_input).resolve()
    total_start = time.time()

    if input_path.is_dir():
        files = []
        extensions = ("*.pdf", "*.png", "*.jpg", "*.jpeg", "*.webp")
        for ext in extensions:
            if args.recursive:
                files.extend(input_path.rglob(ext))
            else:
                files.extend(input_path.glob(ext))
        files = sorted(list(set(files)))
        print(f"Found {len(files)} files in {input_path} (recursive={args.recursive})")
        for f in files:
            process_file(
                str(f),
                model=model_tag,
                scale=args.scale,
                output_dir=target_output_dir,
                keep_header_footer=args.keep_header_footer,
                max_tokens=args.max_tokens,
                context_size=args.context_size,
                pages=args.pages,
                post_crop=args.post_crop,
            )
        total_elapsed = time.time() - total_start
        print(f"\n[✓] Batch completed: {len(files)} file(s) processed in {format_duration(total_elapsed)}.")
    else:
        process_file(
            str(input_path),
            model=model_tag,
            scale=args.scale,
            output_dir=target_output_dir,
            keep_header_footer=args.keep_header_footer,
            max_tokens=args.max_tokens,
            context_size=args.context_size,
            pages=args.pages,
            post_crop=args.post_crop,
        )
        total_elapsed = time.time() - total_start
        print(f"\n[✓] OCR process finished in {format_duration(total_elapsed)}.")


if __name__ == "__main__":
    main()
