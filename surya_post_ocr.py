"""
Surya Post-OCR: Crop Bounding Box Visual Regions & Insert into OCR Results
Optimized for Surya OCR 2 (0.65B) outputs on Windows.

Workflow:
1. Parses Surya OCR results (.md or .html) containing spatial <div data-bbox="x0 y0 x1 y1" data-label="...">.
2. Identifies visual regions (Chemical-Block, Figure, Picture, Image, or elements containing <img> tags).
3. Renders the exact page from the source PDF (or loads source image) on-demand.
4. Crops bounding box regions with optional padding.
5. Saves cropped image assets into a designated folder (default: <stem>_crops/).
6. Generates updated OCR text with image links seamlessly inserted (HTML <img> or Markdown syntax).
"""

import os
import sys
import re
import time
import argparse
import pathlib
from typing import Optional, Tuple, List, Dict

try:
    from PIL import Image
except ImportError:
    print("[-] Pillow is required. Please install it with: pip install pillow")
    sys.exit(1)


# Hardcode default input/source paths here if desired:
# When set, you can run simply: python surya_post_ocr.py
DEFAULT_INPUT_PATH = r""
DEFAULT_SOURCE_PATH = r""
DEFAULT_CROPS_DIR = r""

# Visual labels to automatically crop by default
DEFAULT_VISUAL_LABELS = {
    "chemical-block",
    "figure",
    "picture",
    "image",
    "diagram",
    "chart",
    "graphic",
    "drawing",
    "photo",
}


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration (e.g. '4.2s' or '1m 12.5s')."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}m {secs:.1f}s"


def clean_label_for_filename(label: str) -> str:
    """Sanitize label string for safe filenames."""
    cleaned = re.sub(r"[^\w\-]", "_", label.strip().lower())
    return cleaned if cleaned else "visual"


def clean_caption_for_filename(caption: str) -> str:
    """Sanitize caption string for clean filenames (e.g. 'Chloramphenicol-D5' -> 'chloramphenicol_d5')."""
    s = caption.strip().lower()
    s = re.sub(r"[^\w\-]", "_", s)
    s = re.sub(r"[-_]+", "_", s).strip("_")
    return s if s else "caption"


_RAPID_OCR_INSTANCE = None


def get_micro_ocr():
    """Lazily load RapidOCR if available."""
    global _RAPID_OCR_INSTANCE
    if _RAPID_OCR_INSTANCE is not None:
        return _RAPID_OCR_INSTANCE
    try:
        import logging
        for logger_name in ("RapidOCR", "rapidocr", "ppocr"):
            logging.getLogger(logger_name).setLevel(logging.WARNING)
        from rapidocr import RapidOCR
        _RAPID_OCR_INSTANCE = RapidOCR()
        return _RAPID_OCR_INSTANCE
    except Exception:
        return None


def extract_bottom_caption(crop_image: Image.Image) -> Optional[str]:
    """
    Use lightweight micro-OCR (RapidOCR) to detect and extract bottom caption/label text
    from a cropped figure. Looks for text in the bottom zone of the image, joining horizontal parts.
    """
    ocr_engine = get_micro_ocr()
    if ocr_engine is None:
        return None

    try:
        import numpy as np
        img_arr = np.array(crop_image.convert("RGB"))
        out = ocr_engine(img_arr)
        if not out or out.boxes is None or len(out.boxes) == 0 or not out.txts:
            return None

        W, H = crop_image.size
        candidates = []
        for box, txt, score in zip(out.boxes, out.txts, out.scores):
            txt = txt.strip()
            if not txt or score < 0.45:
                continue
            y_mid = float(np.mean(box[:, 1]))
            x_mid = float(np.mean(box[:, 0]))
            candidates.append({
                "text": txt,
                "y_mid": y_mid,
                "x_mid": x_mid,
                "score": score,
            })

        if not candidates:
            return None

        # Sort candidates by vertical position descending (bottom to top)
        candidates.sort(key=lambda c: c["y_mid"], reverse=True)
        bottom_cand = candidates[0]

        # Only accept if it's sufficiently low in the crop (bottom half: y_mid >= 0.45 * H)
        if bottom_cand["y_mid"] < 0.45 * H:
            return None

        # Gather any other text boxes on the same horizontal baseline (within +-20px vertically)
        baseline_y = bottom_cand["y_mid"]
        same_line_cands = [
            c for c in candidates
            if abs(c["y_mid"] - baseline_y) <= 20
        ]
        # Sort left to right
        same_line_cands.sort(key=lambda c: c["x_mid"])
        caption = " ".join(c["text"] for c in same_line_cands).strip()

        # Clean trailing/leading punctuation
        caption = re.sub(r"^[.,;:_\-\s]+|[.,;:_\-\s]+$", "", caption)
        return caption if len(caption) >= 2 else None
    except Exception:
        return None


def parse_page_range(page_spec: str, total_pages: int) -> set:
    """
    Parse page specifications such as '1-3', '5', '1,3,5', '2-', '-4'.
    Returns a sorted 0-based set of page indices.
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
    return selected if selected else set(range(total_pages))


def find_associated_files(target_path: pathlib.Path) -> Tuple[Optional[pathlib.Path], Optional[pathlib.Path]]:
    """
    Intelligently pair up the OCR file (.md/.html) and the source document (.pdf/.png/.jpg/etc.).
    Given either one, locates the counterpart in the same folder with the same stem.
    """
    target = target_path.resolve()
    if not target.exists():
        return None, None

    ext = target.suffix.lower()
    folder = target.parent
    stem = target.stem

    # If user provided Typhoon layout JSON file
    if ext == ".json":
        clean_stem = stem.replace("-typhoon", "")
        md_candidates = [
            folder / f"{stem}.md",
            folder / f"{stem}-typhoon.md",
            folder / f"{clean_stem}-typhoon.md",
            folder / f"{clean_stem}.md",
        ]
        ocr_file = next((c for c in md_candidates if c.exists()), None)
        source_candidates = [
            folder / f"{stem}.pdf",
            folder / f"{clean_stem}.pdf",
            folder.parent / f"{stem}.pdf",
            folder.parent / f"{clean_stem}.pdf",
            folder / f"{stem}.png",
            folder / f"{clean_stem}.png",
            folder.parent / f"{stem}.png",
            folder.parent / f"{clean_stem}.png",
        ]
        source_file = next((c for c in source_candidates if c.exists()), None)
        return ocr_file or target, source_file

    # If user provided OCR result file (.md, .html, .txt)
    if ext in (".md", ".html", ".htm", ".txt"):
        ocr_file = target
        clean_stem = stem.replace("-typhoon", "").replace("_with_crops", "")
        source_candidates = [
            folder / f"{stem}.pdf",
            folder / f"{clean_stem}.pdf",
            folder.parent / f"{stem}.pdf",
            folder.parent / f"{clean_stem}.pdf",
            folder / f"{stem}.png",
            folder / f"{clean_stem}.png",
            folder.parent / f"{stem}.png",
            folder.parent / f"{clean_stem}.png",
            folder / f"{stem}.jpg",
            folder / f"{clean_stem}.jpg",
            folder.parent / f"{stem}.jpg",
            folder.parent / f"{clean_stem}.jpg",
        ]
        source_file = next((c for c in source_candidates if c.exists()), None)
        return ocr_file, source_file

    # If user provided source document
    if ext in (".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"):
        source_file = target
        clean_stem = stem.replace("-typhoon", "")
        ocr_candidates = [
            folder / f"{stem}.md",
            folder / f"{stem}-typhoon.md",
            folder / f"{clean_stem}-typhoon.md",
            folder / f"{clean_stem}.md",
            folder / f"{stem}.html",
            folder / f"{stem}.json",
            folder / f"{clean_stem}.json",
        ]
        ocr_file = next((c for c in ocr_candidates if c.exists()), None)
        return ocr_file, source_file

    return None, None


def parse_div_attributes(attrs: str) -> Tuple[Optional[List[int]], str]:
    """
    Extract normalized bbox [x0, y0, x1, y1] and data-label from div tag attributes.
    Surya format: data-bbox="x0 y0 x1 y1" (normalized 0-1000).
    """
    bbox = None
    label = "unknown"

    # Match data-bbox="212 83 502 270" or single quotes
    bbox_m = re.search(r'data-bbox=["\']([^"\']+)["\']', attrs, re.IGNORECASE)
    if bbox_m:
        raw_vals = bbox_m.group(1).replace(",", " ").split()
        try:
            bbox = [int(float(v)) for v in raw_vals[:4]]
            if len(bbox) != 4:
                bbox = None
        except ValueError:
            bbox = None

    # Match data-label="..."
    label_m = re.search(r'data-label=["\']([^"\']+)["\']', attrs, re.IGNORECASE)
    if label_m:
        label = label_m.group(1).strip()

    return bbox, label


def crop_bbox_from_image(
    image: Image.Image,
    bbox: List[int],
    padding: int = 6,
    pad_bottom: int = 0,
    auto_expand_bottom_ratio: float = 0.0,
) -> Optional[Image.Image]:
    """
    Convert normalized 0-1000 coordinates to actual pixel coordinates,
    apply padding + downward expansion for bottom captions, and return cropped PIL Image.
    """
    if not bbox or len(bbox) < 4:
        return None

    W, H = image.size
    x0, y0, x1, y1 = bbox

    # Normalized 0-1000 to pixels
    left = int(x0 / 1000.0 * W)
    top = int(y0 / 1000.0 * H)
    right = int(x1 / 1000.0 * W)
    bottom = int(y1 / 1000.0 * H)

    # Calculate downward extension (for captions / chemical names under drawings)
    extra_bottom = pad_bottom + int(auto_expand_bottom_ratio * H)

    # Ensure valid box dimensions
    if right <= left or bottom <= top:
        return None

    # Apply padding and clamp
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(W, right + padding)
    bottom = min(H, bottom + padding + extra_bottom)

    return image.crop((left, top, right, bottom))


def process_surya_post_ocr(
    ocr_file: pathlib.Path,
    source_file: pathlib.Path,
    output_file: Optional[pathlib.Path] = None,
    crops_dir: Optional[pathlib.Path] = None,
    scale: float = 2.0,
    padding: int = 6,
    pad_bottom: int = 0,
    grid: bool = True,
    auto_caption: bool = True,
    target_labels: Optional[List[str]] = None,
    crop_all_images: bool = True,
    crop_all_bboxes: bool = False,
    insert_format: str = "html",
    page_filter: Optional[str] = None,
    inplace: bool = False,
):
    """
    Main post-OCR workflow:
    Parses OCR text, crops bounding boxes from source PDF or image,
    saves cropped image assets, and writes updated OCR text.
    """
    start_time = time.time()
    print(f"\n[+] Surya Post-OCR: Bounding Box Cropper & Document Integrator")
    print(f"  - OCR File   : {ocr_file}")
    print(f"  - Source Doc : {source_file}")

    with open(ocr_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Determine crops directory
    if crops_dir is None:
        crops_dir = ocr_file.parent / f"{ocr_file.stem}_crops"

    try:
        crops_dir.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError) as err:
        fallback_crops_dir = pathlib.Path.cwd() / f"{ocr_file.stem}_crops"
        print(f"  [!] Notice: Cannot write to '{crops_dir}' ({err}).")
        print(f"      Redirecting crops to local folder: {fallback_crops_dir}")
        crops_dir = fallback_crops_dir
        crops_dir.mkdir(parents=True, exist_ok=True)

    print(f"  - Crops Dir  : {crops_dir}")

    # Determine output file
    if inplace:
        final_output_path = ocr_file
    elif output_file:
        final_output_path = output_file
    else:
        desired_path = ocr_file.parent / f"{ocr_file.stem}_with_crops{ocr_file.suffix}"
        # Test if target directory has write permissions
        try:
            test_probe = desired_path.parent / ".write_probe_tmp"
            with open(test_probe, "w", encoding="utf-8") as f:
                f.write("")
            test_probe.unlink(missing_ok=True)
            final_output_path = desired_path
        except (PermissionError, OSError):
            fallback_out = pathlib.Path.cwd() / f"{ocr_file.stem}_with_crops{ocr_file.suffix}"
            print(f"  [!] Notice: Target folder not writable. Output redirected to: {fallback_out}")
            final_output_path = fallback_out

    # Target labels set
    normalized_target_labels = set()
    if target_labels:
        for lbl in target_labels:
            normalized_target_labels.add(lbl.strip().lower())

    # Detect PDF vs Image source
    is_pdf = source_file.suffix.lower() == ".pdf"
    pdf_doc = None
    pdf_backend = None
    total_pages = 1

    if is_pdf:
        try:
            import pypdfium2 as pdfium
            pdf_backend = "pypdfium2"
            pdf_doc = pdfium.PdfDocument(source_file)
            total_pages = len(pdf_doc)
        except ImportError:
            try:
                import fitz
                pdf_backend = "pymupdf"
                pdf_doc = fitz.open(source_file)
                total_pages = len(pdf_doc)
            except ImportError:
                print("[-] Error: Rendering PDF requires pypdfium2 or PyMuPDF.")
                print("    Please install: pip install pypdfium2")
                return

        print(f"  - Source Type: PDF ({total_pages} pages, backend: {pdf_backend}, render scale: {scale}x)")
    else:
        print(f"  - Source Type: Single Image ({source_file.suffix})")

    # Parse page specification if provided
    allowed_page_indices = (
        parse_page_range(page_filter, total_pages) if page_filter else set(range(total_pages))
    )

    # Split document into page sections if marked with "## Page <N>"
    page_split_pattern = re.compile(r"(?m)(^## Page\s+(\d+)\s*$)", re.MULTILINE)
    has_page_headers = bool(page_split_pattern.search(content))

    sections = []
    if has_page_headers:
        # Find all header positions
        matches = list(page_split_pattern.finditer(content))
        last_end = 0
        for i, match in enumerate(matches):
            header_full = match.group(1)
            page_num = int(match.group(2))
            start_pos = match.start()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(content)

            # Keep any pre-page content (e.g. metadata or preamble)
            if i == 0 and start_pos > 0:
                preamble = content[:start_pos]
                sections.append({"page": 0, "header": "", "body": preamble, "is_preamble": True})

            body = content[match.end() : end_pos]
            sections.append(
                {"page": page_num, "header": header_full, "body": body, "is_preamble": False}
            )
    else:
        # Single page / image document
        sections.append({"page": 1, "header": "", "body": content, "is_preamble": False})

    # Regex to match spatial div elements:
    # <div data-bbox="..." data-label="...">...</div>
    div_pattern = re.compile(
        r"(<div\s+([^>]*data-bbox=[\"'][^\"']+[\"'][^>]*)>(.*?)</div>)",
        re.DOTALL | re.IGNORECASE,
    )

    # If content has no <div data-bbox>, check for companion Typhoon layout .json
    if not div_pattern.search(content):
        stem_clean = ocr_file.stem.replace("-typhoon", "")
        json_candidates = [
            ocr_file.parent / f"{ocr_file.stem}.json",
            ocr_file.parent / f"{stem_clean}.json",
        ]
        companion_json = next((j for j in json_candidates if j.exists()), None)
        if companion_json:
            try:
                import json
                with open(companion_json, "r", encoding="utf-8") as jf:
                    json_data = json.load(jf)
                page_figures_map = {}
                if isinstance(json_data, list):
                    for item in json_data:
                        if isinstance(item, dict) and "page" in item:
                            p_num = item["page"]
                            for elem in item.get("elements", []):
                                if elem.get("category") == "figure" and elem.get("bbox"):
                                    page_figures_map.setdefault(p_num, []).append(elem["bbox"])
                        elif isinstance(item, dict) and item.get("category") == "figure" and item.get("bbox"):
                            page_figures_map.setdefault(1, []).append(item["bbox"])

                if page_figures_map:
                    print(f"  [+] Ingested figure bboxes from companion Typhoon layout: {companion_json.name}")
                    for sec in sections:
                        p_num = sec["page"]
                        if p_num in page_figures_map:
                            injected_divs = []
                            for bbox in page_figures_map[p_num]:
                                b_str = " ".join(str(int(x)) for x in bbox)
                                injected_divs.append(f'<div data-bbox="{b_str}" data-label="Figure"><img/></div>')
                            sec["body"] = sec["body"].rstrip() + "\n\n" + "\n".join(injected_divs) + "\n\n"
            except Exception as jerr:
                print(f"  [-] Warning: could not parse companion json: {jerr}")

    total_crops_saved = 0
    pages_with_crops = 0

    # Process section by section
    for section in sections:
        if section.get("is_preamble"):
            continue

        page_num = section["page"]
        page_idx = page_num - 1

        if is_pdf and page_idx not in allowed_page_indices:
            continue

        body_text = section["body"]

        # Check if page body contains any visual candidates before rendering image
        div_matches = list(div_pattern.finditer(body_text))
        candidates = []
        for m in div_matches:
            full_div = m.group(1)
            attrs = m.group(2)
            inner = m.group(3)
            bbox, label = parse_div_attributes(attrs)
            if not bbox:
                continue

            lbl_lower = label.lower()
            should_crop = False

            if crop_all_bboxes:
                should_crop = True
            elif normalized_target_labels and lbl_lower in normalized_target_labels:
                should_crop = True
            elif crop_all_images and ("<img" in inner.lower() or lbl_lower in DEFAULT_VISUAL_LABELS):
                should_crop = True

            if should_crop:
                candidates.append((m, bbox, label, inner))

        if not candidates:
            continue

        # Visual candidates found! Load/render the page image now
        page_img = None
        bitmap = None
        try:
            if is_pdf:
                if pdf_backend == "pypdfium2":
                    p = pdf_doc[page_idx]
                    bitmap = p.render(scale=scale)
                    page_img = bitmap.to_pil()
                else:
                    p = pdf_doc[page_idx]
                    pix = p.get_pixmap(matrix=fitz.Matrix(scale, scale))
                    page_img = Image.open(io.BytesIO(pix.tobytes("png")))
                    pix = None
            else:
                page_img = Image.open(source_file)

            W, H = page_img.size
            print(f"\n  [+] Processing Page {page_num} (Source resolution: {W}x{H}) - {len(candidates)} visual block(s) detected:")

            # 1. Process and crop each candidate
            crop_results = []
            saved_filenames = set()
            for idx, (m, bbox, label, inner) in enumerate(candidates, 1):
                clean_lbl = clean_label_for_filename(label)
                # If label is generic 'figure' and no pad_bottom set, auto-expand downward by 3.2% of page height
                # to cleanly include chemical/figure captions printed beneath drawings
                auto_expand = 0.032 if (pad_bottom == 0 and label.lower() == "figure" and bbox[3] < 960) else 0.0
                cropped = crop_bbox_from_image(
                    page_img,
                    bbox,
                    padding=padding,
                    pad_bottom=pad_bottom,
                    auto_expand_bottom_ratio=auto_expand,
                )
                if cropped is None:
                    continue

                caption = None
                if auto_caption and (label.lower() in DEFAULT_VISUAL_LABELS or crop_all_images):
                    caption = extract_bottom_caption(cropped)

                if caption:
                    clean_cap = clean_caption_for_filename(caption)
                    base_name = f"page_{page_num:02d}_{clean_cap}"
                    crop_filename = f"{base_name}.png"
                    counter = 2
                    while crop_filename in saved_filenames:
                        crop_filename = f"{base_name}_{counter}.png"
                        counter += 1
                else:
                    crop_filename = f"page_{page_num:02d}_crop_{idx:02d}_{clean_lbl}.png"

                saved_filenames.add(crop_filename)
                crop_disk_path = crops_dir / crop_filename
                cropped.save(crop_disk_path)
                total_crops_saved += 1

                try:
                    rel_img_path = os.path.relpath(crop_disk_path, final_output_path.parent).replace("\\", "/")
                except ValueError:
                    rel_img_path = crop_disk_path.as_posix()

                crop_results.append({
                    "match": m,
                    "bbox": bbox,
                    "label": label,
                    "caption": caption,
                    "inner": inner,
                    "rel_path": rel_img_path,
                    "filename": crop_filename,
                    "idx": idx,
                })
                cap_display = f' (caption: "{caption}")' if caption else ""
                print(f"    - Crop #{idx:02d}: [{label}] bbox={bbox} -> {crop_filename}{cap_display}")

            if not crop_results:
                continue

            def render_figure_element(item, in_grid: bool = False) -> str:
                cap = item.get("caption")
                rel_p = item["rel_path"]
                lbl = item["label"]
                b_str = " ".join(str(x) for x in item["bbox"])
                if cap:
                    margin_style = "margin: 4px auto; text-align: center;" if in_grid else "margin: 12px auto; text-align: center;"
                    return (
                        f'<figure style="{margin_style}">\n'
                        f'  <img src="{rel_p}" alt="{cap}" style="max-width: 100%; height: auto;" />\n'
                        f'  <figcaption style="margin-top: 6px;"><b>{cap}</b></figcaption>\n'
                        f'</figure>'
                    )
                else:
                    return f'<div data-bbox="{b_str}" data-label="{lbl}"><img src="{rel_p}" alt="{lbl}" style="max-width: 100%;" /></div>'

            # 2. Check if candidates can be grouped into side-by-side rows (e.g. 2-column layout)
            # Sort by reading order: top-to-bottom (y), then left-to-right (x)
            sorted_items = sorted(crop_results, key=lambda c: (c["bbox"][1], c["bbox"][0]))
            has_pairs = False
            for i in range(len(sorted_items) - 1):
                b1 = sorted_items[i]["bbox"]
                b2 = sorted_items[i + 1]["bbox"]
                if abs(b1[1] - b2[1]) <= 65 and b1[0] < b2[0]:
                    has_pairs = True
                    break

            # If candidates are contiguous in body text and form a multi-column layout
            start_indices = [item["match"].start() for item in crop_results]
            end_indices = [item["match"].end() for item in crop_results]
            min_start = min(start_indices)
            max_end = max(end_indices)
            span_text = body_text[min_start:max_end]
            # Verify if intermediate text is only whitespace/newlines/divs
            non_div_residual = re.sub(r"<div[^>]*>.*?</div>", "", span_text, flags=re.DOTALL).strip()
            is_contiguous_block = len(non_div_residual) == 0

            if grid and has_pairs and is_contiguous_block and insert_format != "markdown":
                # Build clean 2-column table
                rows = []
                i = 0
                while i < len(sorted_items):
                    item = sorted_items[i]
                    bbox = item["bbox"]
                    if i + 1 < len(sorted_items):
                        next_item = sorted_items[i + 1]
                        next_bbox = next_item["bbox"]
                        if abs(bbox[1] - next_bbox[1]) <= 65 and bbox[0] < next_bbox[0]:
                            rows.append([item, next_item])
                            i += 2
                            continue
                    rows.append([item])
                    i += 1

                grid_lines = ['\n<table border="0" style="width: 100%; border-collapse: collapse; margin: 16px auto;">']
                for r in rows:
                    grid_lines.append('  <tr>')
                    if len(r) == 2:
                        cell1 = render_figure_element(r[0], in_grid=True)
                        cell2 = render_figure_element(r[1], in_grid=True)
                        grid_lines.append(f'    <td align="center" style="width: 50%; vertical-align: top; padding: 8px;">{cell1}</td>')
                        grid_lines.append(f'    <td align="center" style="width: 50%; vertical-align: top; padding: 8px;">{cell2}</td>')
                    else:
                        cell = render_figure_element(r[0], in_grid=True)
                        grid_lines.append(f'    <td align="center" colspan="2" style="vertical-align: top; padding: 8px;">{cell}</td>')
                    grid_lines.append('  </tr>')
                grid_lines.append('</table>\n')
                replacement_table = "\n".join(grid_lines)

                new_body = body_text[:min_start] + replacement_table + body_text[max_end:]
            else:
                # Sequential individual replacements (from back to front)
                new_body = body_text
                for item in reversed(crop_results):
                    m = item["match"]
                    bbox = item["bbox"]
                    label = item["label"]
                    cap = item.get("caption")
                    inner = item["inner"]
                    rel_img_path = item["rel_path"]
                    crop_idx = item["idx"]

                    start_div = m.start()
                    end_div = m.end()

                    if insert_format == "markdown":
                        if cap:
                            replacement = f"\n\n![{cap}]({rel_img_path})\n*<b>{cap}</b>*\n\n"
                        else:
                            alt_text = f"{label} (Page {page_num}, crop {crop_idx})"
                            replacement = f"\n\n![{alt_text}]({rel_img_path})\n\n"
                    elif insert_format == "both":
                        fig_html = render_figure_element(item, in_grid=False)
                        alt = cap if cap else label
                        replacement = f"\n{fig_html}\n\n![{alt}]({rel_img_path})\n"
                    else:
                        if cap:
                            replacement = f"\n{render_figure_element(item, in_grid=False)}\n"
                        else:
                            if "<img" in inner.lower():
                                new_inner = re.sub(
                                    r"<img\s*\/?>",
                                    f'<img src="{rel_img_path}" alt="{label}" />',
                                    inner,
                                    flags=re.IGNORECASE,
                                )
                            else:
                                clean_inner = inner.strip()
                                if clean_inner:
                                    new_inner = f'{clean_inner}\n<img src="{rel_img_path}" alt="{label}" />'
                                else:
                                    new_inner = f'<img src="{rel_img_path}" alt="{label}" />'
                            replacement = f"<div {m.group(2)}>{new_inner}</div>"

                    new_body = new_body[:start_div] + replacement + new_body[end_div:]

            section["body"] = new_body
            pages_with_crops += 1

        finally:
            if bitmap is not None and hasattr(bitmap, "close"):
                bitmap.close()
            if page_img is not None and hasattr(page_img, "close"):
                page_img.close()

    if is_pdf and pdf_doc is not None:
        try:
            pdf_doc.close()
        except Exception:
            pass

    # Reconstruct the document
    reconstructed_text = ""
    for section in sections:
        if section.get("is_preamble"):
            reconstructed_text += section["body"]
        else:
            if section["header"]:
                reconstructed_text += f"{section['header']}\n"
            reconstructed_text += section["body"]

    # Write output file
    with open(final_output_path, "w", encoding="utf-8") as f:
        f.write(reconstructed_text)

    elapsed = time.time() - start_time
    print(f"\n[✓] Surya Post-OCR Finished in {format_duration(elapsed)}:")
    print(f"  - Total Visual Crops Saved : {total_crops_saved}")
    print(f"  - Pages with Visual Elements: {pages_with_crops}")
    print(f"  - Crops Directory           : {crops_dir}")
    print(f"  - Output Document           : {final_output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Surya Post-OCR: Crop bounding box regions from source PDF/image and insert into OCR text results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 1. Automatic pairing (detects method-amp.pdf from method-amp.md in same folder):
  python surya_post_ocr.py "E:\\docs\\method-amp.md"

  # 2. Specify source PDF or image explicitly:
  python surya_post_ocr.py "E:\\docs\\method-amp.md" --source "E:\\docs\\method-amp.pdf"

  # 3. Output pure Markdown images instead of HTML:
  python surya_post_ocr.py "E:\\docs\\method-amp.md" --format markdown

  # 4. Target specific labels (e.g., Chemical-Block, Table):
  python surya_post_ocr.py "E:\\docs\\method-amp.md" --labels Chemical-Block Table

  # 5. Overwrite the OCR file in-place:
  python surya_post_ocr.py "E:\\docs\\method-amp.md" --inplace
""",
    )

    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Path to the OCR result file (.md, .html) or source document (.pdf, image).",
    )
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        default=None,
        help="Path to the source document (PDF or image). Optional if matching stem exists in the same folder.",
    )
    parser.add_argument(
        "--ocr",
        type=str,
        default=None,
        help="Path to the OCR text/markdown file. Optional if matching stem exists in the same folder.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path for the output markdown/html file (defaults to <stem>_with_crops.md).",
    )
    parser.add_argument(
        "--crops-dir",
        "-c",
        type=str,
        default=None,
        help="Directory to store cropped images (defaults to <stem>_crops in the document folder).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="PDF rendering scale factor (default: 2.0x = ~144 DPI for crisp crops).",
    )
    parser.add_argument(
        "--padding",
        "-p",
        type=int,
        default=6,
        help="Pixel margin padding around the bounding box (default: 6px).",
    )
    parser.add_argument(
        "--labels",
        "-l",
        nargs="+",
        default=None,
        help="Specific data-labels to crop (e.g. Chemical-Block Figure Table). Defaults to all visual elements.",
    )
    parser.add_argument(
        "--all-bboxes",
        action="store_true",
        help="Crop every element that has a data-bbox attribute (e.g., for layout dataset generation).",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["html", "markdown", "both"],
        default="html",
        help="Format for inserting image links in the OCR text: 'html' (updates <img src='...'> inside div), 'markdown' (![alt](path)), or 'both' (default: html).",
    )
    parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help="Page range to process for PDFs (e.g. '10', '1-5', '1,3,10'). Defaults to all pages.",
    )
    parser.add_argument(
        "--pad-bottom",
        type=int,
        default=0,
        help="Extra downward pixel padding to include captions/labels below figures (default: 0 = auto-expand for figure drawings).",
    )
    parser.add_argument(
        "--no-grid",
        dest="grid",
        action="store_false",
        default=True,
        help="Disable automatic side-by-side 2-column table layout for horizontal figure pairs.",
    )
    parser.add_argument(
        "--no-caption",
        dest="auto_caption",
        action="store_false",
        default=True,
        help="Disable automatic caption detection via micro-OCR (RapidOCR) and <figure>/<figcaption> generation.",
    )
    parser.add_argument(
        "--inplace",
        "-i",
        action="store_true",
        help="Overwrite the input OCR file in-place instead of creating a new file.",
    )

    args = parser.parse_args()

    # Determine input files
    ocr_path = None
    source_path = None

    if args.ocr:
        ocr_path = pathlib.Path(args.ocr).resolve()
    if args.source:
        source_path = pathlib.Path(args.source).resolve()

    target_input = args.input or (DEFAULT_INPUT_PATH if DEFAULT_INPUT_PATH.strip() else None)

    if target_input:
        in_p = pathlib.Path(target_input).resolve()
        if in_p.is_dir():
            # Batch process directory
            ocr_files = sorted(list(in_p.glob("*.md")) + list(in_p.glob("*.html")))
            processed_count = 0
            for f in ocr_files:
                # Skip previously generated with_crops files
                if "_with_crops" in f.name:
                    continue
                pair_ocr, pair_src = find_associated_files(f)
                if pair_ocr and pair_src:
                    process_surya_post_ocr(
                        ocr_file=pair_ocr,
                        source_file=pair_src,
                        output_file=pathlib.Path(args.output).resolve() if args.output else None,
                        crops_dir=pathlib.Path(args.crops_dir).resolve() if args.crops_dir else None,
                        scale=args.scale,
                        padding=args.padding,
                        pad_bottom=args.pad_bottom,
                        grid=args.grid,
                        auto_caption=args.auto_caption,
                        target_labels=args.labels,
                        crop_all_images=True,
                        crop_all_bboxes=args.all_bboxes,
                        insert_format=args.format,
                        page_filter=args.pages,
                        inplace=args.inplace,
                    )
                    processed_count += 1
            print(f"\n[✓] Directory batch finished: {processed_count} document(s) processed.")
            return

        auto_ocr, auto_source = find_associated_files(in_p)
        if ocr_path is None:
            ocr_path = auto_ocr
        if source_path is None:
            source_path = auto_source

    if source_path is None and DEFAULT_SOURCE_PATH.strip():
        source_path = pathlib.Path(DEFAULT_SOURCE_PATH).resolve()

    if not ocr_path or not ocr_path.exists():
        print("[-] Error: Could not find OCR text file (.md or .html).")
        print("    Usage: python surya_post_ocr.py <path_to_ocr.md> [--source <path_to_doc.pdf>]")
        print("    Tip: You can set DEFAULT_INPUT_PATH at line 31 in surya_post_ocr.py to omit the path.")
        sys.exit(1)

    if not source_path or not source_path.exists():
        print(f"[-] Error: Could not find source document for: {ocr_path.name}")
        print(f"    Expected matching PDF or image in: {ocr_path.parent}")
        print("    Specify explicitly with: --source <path_to_source_doc>")
        sys.exit(1)

    output_path = pathlib.Path(args.output).resolve() if args.output else None
    crops_path = (
        pathlib.Path(args.crops_dir).resolve()
        if args.crops_dir
        else (pathlib.Path(DEFAULT_CROPS_DIR).resolve() if DEFAULT_CROPS_DIR.strip() else None)
    )

    process_surya_post_ocr(
        ocr_file=ocr_path,
        source_file=source_path,
        output_file=output_path,
        crops_dir=crops_path,
        scale=args.scale,
        padding=args.padding,
        pad_bottom=args.pad_bottom,
        grid=args.grid,
        auto_caption=args.auto_caption,
        target_labels=args.labels,
        crop_all_images=True,
        crop_all_bboxes=args.all_bboxes,
        insert_format=args.format,
        page_filter=args.pages,
        inplace=args.inplace,
    )


if __name__ == "__main__":
    main()
