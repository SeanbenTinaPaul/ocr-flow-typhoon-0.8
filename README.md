# Hands-on Guide: Local Small OCR Models on GTX 1050 (3GB Laptop)

This guide walks newcomers step-by-step through running, managing, and utilizing local document OCR models (**Typhoon OCR 2 0.8B** and **Surya OCR 2 0.65B**) using **Ollama** on a resource-constrained laptop (NVIDIA GeForce GTX 1050 Mobile with 3GB VRAM).

---

## ⚠️ Fundamental Rule: Single-Model GPU Execution

> [!IMPORTANT]
> **Why are there two models, and why must they NEVER load simultaneously?**
>
> 1. **The Models:**
>    - **`typhoon-ocr:0.8b` (0.75B params, 1.7 GB):** Specialized in Thai & English full-page parsing. Outputs reading-order layout elements with normalized bounding boxes `[x1, y1, x2, y2]`, HTML tables, LaTeX formulas, and Markdown.
>    - **`surya-ocr:0.65b` (0.63B params, 1.5 GB):** Specialized in multilingual document layout, table recognition, and semantic HTML `<div>` blocks with spatial tags (`data-bbox` and `data-label`).
>
> 2. **The Hardware Constraint:**
>    - Your GTX 1050 Mobile GPU has strictly **3,072 MB (3 GB) of VRAM**.
>    - One model with its weights (~1.5–1.7 GB), recurrent KV-cache (~390 MB for 16k context), and CUDA compute buffers takes **~2.1 GB of VRAM**.
>    - If both models were loaded into VRAM at once, they would require **~4.2 GB of VRAM**, immediately triggering GPU Out-Of-Memory (OOM) or freezing the machine into slow CPU swap.
>
> 3. **How It Is Handled:**
>    - The Python runner (`run_ocr_0.8b.py`) enforces **`keep_alive: 0`**.
>    - Ollama loads the selected model into VRAM, processes your document, and **immediately purges it from VRAM back to 0 MB**.
>    - You can freely choose between `--model typhoon` or `--model surya`, but they will **always run one-by-one sequentially**.

---

## 1. Storage & Environment Setup (Preserving Drive C:)

Ollama defaults to storing model blobs in `C:\Users\<user>\.ollama\models`. Because SSD `C:` space is limited, the storage location on this laptop is redirected to Drive `E:`.

### Setting the Environment Variable

#### PowerShell:
```powershell
# Set permanently for the current user
[System.Environment]::SetEnvironmentVariable("OLLAMA_MODELS", "E:\Download_hub\ollama_models", "User")

# Set for current terminal session
$env:OLLAMA_MODELS = "E:\Download_hub\ollama_models"
```

#### Command Prompt (CMD):
```cmd
:: Set permanently for the current user
setx OLLAMA_MODELS "E:\Download_hub\ollama_models"

:: Set for current terminal session
set OLLAMA_MODELS=E:\Download_hub\ollama_models
```

---

## 2. Managing Ollama from the Command Line

### A. Starting the Ollama Server
On Windows laptops where auto-start is disabled, start the server before running inference:

#### PowerShell (Hidden Background Process)
Runs headless in the background without keeping an extra terminal open:
```powershell
$env:OLLAMA_MODELS = "E:\Download_hub\ollama_models"
Start-Process "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" -ArgumentList "serve" -WindowStyle Hidden
```

#### Command Prompt (CMD - Foreground with Live Logs)
Runs in the foreground showing real-time server logs (press `Ctrl+C` to stop):
```cmd
set OLLAMA_MODELS=E:\Download_hub\ollama_models
ollama serve
```

*(Or run headless in background via CMD):*
```cmd
set OLLAMA_MODELS=E:\Download_hub\ollama_models
start /B "" "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve
```
*Wait 2–3 seconds for the server to initialize on `http://127.0.0.1:11434`.*

### B. Essential Ollama Commands

| Goal | Command | Description |
| :--- | :--- | :--- |
| **Check Installed Models** | `ollama list` | Shows all models currently stored in `E:\Download_hub\ollama_models`. |
| **Check Active VRAM** | `ollama ps` | Shows models currently occupying GPU VRAM. **Should be empty `{}` when idle.** |
| **Inspect Model Specs** | `ollama show typhoon-ocr:0.8b` | Displays architecture, context window, and system template. |
| **Delete a Model** | `ollama rm <model-name>` | Deletes model weights from disk to free up space. |

---

## 3. How the Modelfiles Are Built (Modern Ollama Dual-`FROM` Syntax)

Modern Ollama (v0.34+) removed the deprecated `ADAPTER` directive. Vision-Language Models (VLMs) with separate vision projectors (`mmproj`) are now constructed using **dual `FROM` lines**:

### Typhoon OCR 2 (0.8B) Modelfile
```dockerfile
FROM "E:/Download_hub/hf_cache/hub/models--typhoon-ai--typhoon-ocr2-0.8b-GGUF/snapshots/e79d5592515082cd42001e7876cfffdb2b45fede/typhoon-ocr2-0.8b-f16.gguf"
FROM "E:/Download_hub/hf_cache/hub/models--typhoon-ai--typhoon-ocr2-0.8b-GGUF/snapshots/e79d5592515082cd42001e7876cfffdb2b45fede/mmproj-typhoon-ocr2-0.8b-f16.gguf"

# 16k context window gives ~14k tokens for output after ~2.4k vision image tokens
PARAMETER num_ctx 16384
PARAMETER num_gpu 999
PARAMETER temperature 0.0
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""

SYSTEM """- Extract layout information from the provided PDF image.
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
```

To (re)create the model in Ollama:

**PowerShell:**
```powershell
$env:OLLAMA_MODELS = "E:\Download_hub\ollama_models"
ollama create typhoon-ocr:0.8b -f "E:\Download_hub\ocr_scripts\Modelfile.typhoon-ocr2"
```

**Command Prompt (CMD):**
```cmd
set OLLAMA_MODELS=E:\Download_hub\ollama_models
ollama create typhoon-ocr:0.8b -f "E:\Download_hub\ocr_scripts\Modelfile.typhoon-ocr2"
```

---

## 4. Hands-on OCR Execution with `run_ocr_0.8b.py`

The Python script [`run_ocr_0.8b.py`](./run_ocr_0.8b.py) uses Python's standard library (`urllib.request`, `base64`, `json`, `re`) for image processing. No heavy packages are required for images.

### Prerequisites
* **Python 3.10+** (Available at `C:\Python314\python.exe` or `E:\uv\bin\uv.exe`).
* *(For PDF rendering)*: Uses `pypdfium2` (already installed on this rig) with automatic fallback to `PyMuPDF`.

> [!TIP]
> **Hardcoding Default Input and Output Paths:**
> You can set `DEFAULT_INPUT_PATH` and `DEFAULT_OUTPUT_DIR` at lines 32–36 in [`run_ocr_0.8b.py`](./run_ocr_0.8b.py):
> ```python
> DEFAULT_INPUT_PATH = r"E:\images\my_doc.png"
> DEFAULT_OUTPUT_DIR = r"E:\docs\ocr_results"   # (Leave r"" to save in the input file's folder)
> ```
> Once set, you can run OCR without re-typing any paths:
> ```powershell
> python E:\Download_hub\ocr_scripts\run_ocr_0.8b.py --model surya
> python E:\Download_hub\ocr_scripts\run_ocr_0.8b.py --model typhoon
> ```

---

### Step-by-Step Examples

#### Example 1: Run Typhoon OCR 2 (0.8B) on an Image
This parses Thai or English documents, extracts tables and LaTeX math, and generates both `.md` and `.json`:
```powershell
python E:\Download_hub\ocr_scripts\run_ocr_0.8b.py "E:\images\invoice.png" --model typhoon
```
**Outputs generated in the same directory:**
- `invoice.md`: Clean, human-readable Markdown text with HTML tables and LaTeX formulas.
- `invoice.json`: Raw structured layout list containing bounding boxes `[x1, y1, x2, y2]` and element categories.

#### Example 2: Run Surya OCR (0.65B) on an Image
This generates spatial HTML with bounding boxes:
```powershell
python E:\Download_hub\ocr_scripts\run_ocr_0.8b.py "E:\images\contract.png" --model surya
```

**Command Breakdown:**
* `python`: Invokes the Python interpreter (3.10+).
* `E:\Download_hub\ocr_scripts\run_ocr_0.8b.py`: Path to the runner script.
* `"E:\images\contract.png"`: Input target path (image, multi-page PDF, or folder).
* `--model surya`: Selects the Surya OCR 2 (0.65B) model. It automatically prompts Ollama to load `surya-ocr:0.65b` into GPU VRAM on-demand, format with spatial HTML tags, and immediately flush VRAM back to 0 MB upon completion.

**Output generated:**
- `contract.html`: Semantic HTML elements tagged with `data-bbox="x0 y0 x1 y1"` and `data-label="Section-Header|Table|Text"`.

#### Example 3: Process a Multi-Page PDF
```powershell
# OCR all pages of a PDF:
python E:\Download_hub\ocr_scripts\run_ocr_0.8b.py "E:\docs\annual_report.pdf" --model typhoon --scale 2.0

# OCR only specific pages (e.g., pages 1 through 3, or page 5):
python E:\Download_hub\ocr_scripts\run_ocr_0.8b.py "E:\docs\annual_report.pdf" --model typhoon --pages 1-3
```
*Notes on Multi-page PDF Processing:*
- `--scale 2.0` renders PDF pages at ~144 DPI (retina clarity for small fonts and dense tables).
- Multi-page processing uses session keep-alive (`keep_alive: "5m"`) across pages so the model stays warm in VRAM, and purges to 0 MB upon finishing the final page.
- Progress is written incrementally to `.md` and `.json` after each page, preserving completed work even if interrupted.
- Detailed timing reports: Displays elapsed time per page, total document processing duration, and average seconds-per-page upon finishing.

#### Example 4: Batch Process a Directory of Documents (with Subfolders)
```powershell
python E:\Download_hub\ocr_scripts\run_ocr_0.8b.py "E:\docs\scanned_folder" --model typhoon --recursive --output-dir "E:\docs\ocr_results"
```

---

## 5. Post-OCR: Cropping Visual BBoxes & Embedding Images (`surya_post_ocr.py`)

Whether running **Surya OCR** (`--model surya`) or **Typhoon OCR 2** (`--model typhoon`), visual regions are tagged with normalized bounding boxes:
- **Surya:** Outputs `<div data-bbox="x0 y0 x1 y1" data-label="Chemical-Block|Figure"><img/></div>`.
- **Typhoon:** Outputs `<div data-bbox="x0 y0 x1 y1" data-label="Figure"><img/></div>` in `.md`, and full element trees in `.json`.

The post-processor [`surya_post_ocr.py`](./surya_post_ocr.py) universally processes either model's output:
1. **Scans** the OCR text for visual regions (`Figure`, `Chemical-Block`, `Picture`, `Image`, or tags containing `<img>`).
   *(If an older Typhoon `.md` file without tags is provided, it automatically ingests the figure bboxes from the companion `.json`!)*
2. **Renders** the corresponding page from the source PDF (or loads the source image).
3. **Crops** the bounding box coordinates (`x0 y0 x1 y1`, normalized 0–1000) with configurable pixel padding.
4. **Saves** the cropped images into a designated asset folder (`<stem>_crops/`).
5. **Inserts** relative image links directly into the OCR Markdown or HTML file (`<stem>_with_crops.md`), making images viewable in VS Code, Obsidian, Typora, or GitHub!

### Usage Examples

#### Example 1: Automatic Detection (Typhoon or Surya)
```powershell
# For Typhoon OCR output:
python E:\Download_hub\ocr_scripts\surya_post_ocr.py "E:\docs\method-amp-typhoon.md"

# For Surya OCR output:
python E:\Download_hub\ocr_scripts\surya_post_ocr.py "E:\docs\method-amp.md"
```
*Outputs:*
- `E:\docs\method-amp_crops\page_10_chloramphenicol.png` (auto-detected caption from figure label).
- `E:\docs\method-amp_with_crops.md` with semantic `<figure>` + `<figcaption>` inserted into the reading flow (or 2-column table grid for paired structures).

#### Example 2: Convert to Standard Markdown Images (`--format markdown`)
Replaces the HTML `<div>` blocks with standard Markdown `![label](path)` syntax:
```powershell
python E:\Download_hub\ocr_scripts\surya_post_ocr.py "E:\docs\method-amp.md" --format markdown
```

#### Example 3: Crop Specific Labels (e.g., Figures and Tables)
```powershell
python E:\Download_hub\ocr_scripts\surya_post_ocr.py "E:\docs\report.md" --labels Figure Table Chemical-Block
```

#### Example 4: Process Specific PDF Pages
```powershell
python E:\Download_hub\ocr_scripts\surya_post_ocr.py "E:\docs\method-amp.md" --pages 10
```

#### Example 5: Batch Process an Entire Directory
```powershell
python E:\Download_hub\ocr_scripts\surya_post_ocr.py "E:\docs\lab_reports"
```

### Surya Post-OCR Options Reference

| Argument | Choices / Default | Description |
| :--- | :--- | :--- |
| `input` | *(Required)* Path | OCR file (`.md`, `.html`), source document, or directory. |
| `--source`, `-s` | Path | Path to original PDF or image (auto-detected if omitted). |
| `--output`, `-o` | Path | Output file path (defaults to `<stem>_with_crops.md`). |
| `--crops-dir`, `-c`| Path | Folder for cropped images (defaults to `<stem>_crops/`). |
| `--format`, `-f` | `html` *(default)* \| `markdown` \| `both` | Image link format inserted into the text. |
| `--scale` | Float (default: `2.0`) | Rendering resolution for PDF crops (`2.0` = ~144 DPI for crisp clarity). |
| `--padding`, `-p` | Integer (default: `6`) | Pixel margin padding around cropped bounding boxes. |
| `--pad-bottom` | Integer (default: `0`) | Extra downward pixel padding (default `0` = auto-expands 3.2% for figures). |
| `--no-grid` | Flag (default: on) | Disable 2-column side-by-side table grouping for horizontal pairs. |
| `--no-caption` | Flag (default: on) | Disable micro-OCR caption detection (`RapidOCR`) and `<figure>/<figcaption>` blocks. |
| `--labels`, `-l` | List of strings | Specific `data-label`s to crop (e.g. `Chemical-Block Figure`). |
| `--all-bboxes` | Flag (default: off) | Crop every element with a `data-bbox` (useful for dataset building). |
| `--pages` | String (default: all) | Filter specific PDF pages (e.g. `10`, `1-5`). |
| `--inplace`, `-i` | Flag (default: off) | Overwrite the original OCR file in-place. |

---

## 6. Command-Line Options Reference (OCR Runner `run_ocr_0.8b.py`)

| Argument | Choices / Default | Description |
| :--- | :--- | :--- |
| `input` | *(Required)* File or folder path | Image (`.png`, `.jpg`, `.webp`), PDF (`.pdf`), or directory. |
| `--model` | `typhoon` *(default)* \| `surya` | Which model to load into GPU VRAM for the job. |
| `--pages`, `-p` | String (default: all) | Specific page range to OCR for PDFs (e.g., `1-3`, `5`, `1,3,5`, `2-`). |
| `--recursive`, `-r` | Flag (default: off) | Recursively scan subdirectories for documents. |
| `--scale` | Float (default: `2.0`) | PDF rendering scale multiplier (`2.0` = ~144 DPI, optimal for OCR). |
| `--output-dir`, `-o` | Path (default: input folder) | Custom destination folder for `.md`, `.json`, or `.html` outputs. |
| `--max-tokens` | Integer (default: `4096`) | Maximum new tokens to generate per page. |
| `--context-size` | Integer (default: `16384`) | KV context window allocated in VRAM (leave at `16384` for full pages). |
| `--keep-header-footer`| Flag (default: off) | By default headers and footers are omitted in `.md`; pass this to keep them. |
| `--post-crop` | Flag (default: off) | Automatically trigger post-OCR cropping & integration (`surya_post_ocr.py`) with micro-OCR `<figure>/<figcaption>` blocks immediately after OCR finishes. |

---

## 7. Verification & Troubleshooting

### Q1: How do I know the GPU is not out of memory?
Run this command in another PowerShell tab:
```powershell
ollama ps
```
- **During inference:** Shows 1 active model using ~1.8–2.1 GB VRAM.
- **After inference finishes:** Returns `{}` (empty), confirming VRAM was completely released.

### Q2: Why did `ollama create` give "Error: LoRA adapters are no longer supported"?
You are likely using an old Modelfile with an `ADAPTER` line. Replace `ADAPTER "path/to/mmproj.gguf"` with a second `FROM "path/to/mmproj.gguf"` line as shown in Section 3.

### Q3: Why does OCR stop mid-sentence on complex tables?
Check your context window (`num_ctx`). Images take up ~2,200 to 2,500 input tokens. If `num_ctx` is only 4096, only ~1,600 tokens are left for output. Keep `--context-size 16384` to guarantee full-page completion.
