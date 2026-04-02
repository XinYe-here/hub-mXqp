import argparse
import base64
import mimetypes
import os
from pathlib import Path

import fitz
from openai import OpenAI


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DEFAULT_PDF = PROJECT_DIR / "Week10-多模态大模型.pdf"
DEFAULT_MODEL = "qwen-vl-plus"
DEFAULT_PROMPT = (
    "请解析这份 PDF 的第一页内容。"
    "请用中文输出：1. 页面主题；2. 关键信息摘要；3. 页面里能识别出的标题、要点或版面元素。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use FE8 OpenAI-compatible Qwen-VL to parse the first page of a local PDF."
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_PDF,
        help=f"Path to local PDF. Default: {DEFAULT_PDF}",
    )
    parser.add_argument(
        "--page",
        type=int,
        default=1,
        help="1-based page number. Default: 1",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Prompt sent to the model.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Render DPI for the PDF page image. Default: 200",
    )
    return parser.parse_args()


def render_pdf_page_to_png(pdf_path: Path, page_number: int, dpi: int) -> bytes:
    if page_number < 1:
        raise ValueError("Page number must be >= 1.")

    # Open the PDF locally and render only the requested page to a PNG in memory.
    doc = fitz.open(pdf_path)
    try:
        if page_number > len(doc):
            raise ValueError(f"PDF only has {len(doc)} page(s), but got page={page_number}.")

        page = doc.load_page(page_number - 1)
        # Higher DPI keeps more text/layout detail for the vision model.
        matrix = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        return pix.tobytes("png")
    finally:
        doc.close()


def to_data_url(image_bytes: bytes, filename: str) -> str:
    # The FE8 OpenAI-compatible endpoint accepts image input as a base64 data URL.
    mime_type = mimetypes.guess_type(filename)[0] or "image/png"
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{image_base64}"


def build_client() -> OpenAI:
    # Use the FE8 proxy endpoint and key from environment variables.
    base_url = os.getenv("FE8_BASE_URL") or "https://api.fe8.cn/v1"
    api_key = os.getenv("FE8_API_KEY")

    if not api_key:
        raise EnvironmentError("FE8_API_KEY is not set.")

    return OpenAI(api_key=api_key, base_url=base_url)


def main() -> None:
    args = parse_args()
    pdf_path = args.pdf.resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_dir = SCRIPT_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"PDF: {pdf_path}")
    print(f"Page: {args.page}")
    print(f"Model: {args.model}")

    # Step 1: turn the target PDF page into an image file.
    image_bytes = render_pdf_page_to_png(pdf_path, args.page, args.dpi)
    page_image_path = output_dir / f"{pdf_path.stem}_page_{args.page}.png"
    page_image_path.write_bytes(image_bytes)
    print(f"Rendered page image: {page_image_path}")

    # Step 2: call the multimodal model with image + text prompt.
    client = build_client()
    image_url = to_data_url(image_bytes, page_image_path.name)

    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                    {
                        "type": "text",
                        "text": args.prompt,
                    },
                ],
            }
        ],
        max_tokens=1200,
    )

    # Some OpenAI-compatible services may return either a plain string or structured parts.
    result = response.choices[0].message.content
    if isinstance(result, list):
        result_text = "\n".join(
            part.get("text", str(part)) if isinstance(part, dict) else str(part)
            for part in result
        )
    else:
        result_text = str(result)

    # Step 3: save the parsed text so the result can be reused directly in homework.
    output_text_path = output_dir / f"{pdf_path.stem}_page_{args.page}_parsed.txt"
    output_text_path.write_text(result_text, encoding="utf-8")

    print("\nParsed result:\n")
    print(result_text)
    print(f"\nSaved text output: {output_text_path}")


if __name__ == "__main__":
    main()
