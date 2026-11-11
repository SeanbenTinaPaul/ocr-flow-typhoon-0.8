<#
.SYNOPSIS
    Run local small OCR models (Surya OCR 0.65B or Typhoon OCR 0.8B) via Ollama on GTX 1050 (3GB VRAM).

.DESCRIPTION
    Sends an image to the local Ollama instance with optimal prompt templates,
    spatial bounding boxes, and layout extraction.
    Enforces SINGLE-MODEL execution with immediate memory eviction (keep_alive: 0)
    and an expanded 16k context window for complete document capture.

.PARAMETER ImagePath
    Path to the input image (.png, .jpg, .jpeg, etc.)

.PARAMETER Model
    Either 'surya' (default, HTML layout + bounding boxes) or 'typhoon' (Thai/English JSON elements + bounding boxes)

.PARAMETER MaxTokens
    Max tokens to generate (defaults to 4096)

.PARAMETER ContextSize
    Context window size (defaults to 16384, leaving ~14k tokens for generation after ~2.4k vision tokens)

.EXAMPLE
    .\run_ocr.ps1 -ImagePath "E:\images\document.png" -Model surya
    .\run_ocr.ps1 -ImagePath "E:\images\thai_doc.png" -Model typhoon
#>

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$ImagePath,

    [Parameter(Mandatory=$false, Position=1)]
    [ValidateSet("surya", "typhoon")]
    [string]$Model = "surya",

    [Parameter(Mandatory=$false)]
    [int]$MaxTokens = 4096,

    [Parameter(Mandatory=$false)]
    [int]$ContextSize = 16384
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $ImagePath)) {
    Write-Error "Image file not found: $ImagePath"
    exit 1
}

# Resolve full path and encode to Base64
$resolvedPath = (Resolve-Path $ImagePath).Path
$imgBytes = [System.IO.File]::ReadAllBytes($resolvedPath)
$b64 = [System.Convert]::ToBase64String($imgBytes)

# Choose model tag and prompt template
if ($Model -eq "surya") {
    $modelTag = "surya-ocr:0.65b"
    $prompt = "OCR this image to HTML. Each block is a div with data-label and data-bbox (x0 y0 x1 y1, normalized 0-1000)."
} else {
    $modelTag = "typhoon-ocr:0.8b"
    $prompt = @"
- Extract layout information from the provided PDF image.
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
- Final output must be a single JSON object.
"@
}

# Single-model discipline: keep_alive = 0 immediately frees VRAM after the request finishes
$body = @{
    model = $modelTag
    messages = @(
        @{
            role = "user"
            content = $prompt
            images = @($b64)
        }
    )
    stream = $false
    keep_alive = 0
    options = @{
        num_ctx = $ContextSize
        num_predict = $MaxTokens
        temperature = 0.0
    }
} | ConvertTo-Json -Depth 5

Write-Host "Invoking $modelTag on: $(Split-Path $resolvedPath -Leaf) (num_ctx: $ContextSize, keep_alive: 0)..." -ForegroundColor Cyan

try {
    $response = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:11434/api/chat" -Body $body -ContentType "application/json" -TimeoutSec 180
    Write-Output $response.message.content
} catch {
    Write-Error "Ollama inference failed. Ensure Ollama is running (`ollama serve`). Details: $_"
}
