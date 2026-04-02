import argparse
from pathlib import Path

import clip
import torch
from PIL import Image


DEFAULT_MODEL = "ViT-B/32"
DEFAULT_LABELS = ["dog", "cat", "bird", "car", "person"]
SCRIPT_DIR = Path(__file__).resolve().parent
# The first run downloads CLIP weights here, later runs reuse the cache.
MODEL_CACHE_DIR = SCRIPT_DIR / "clip_cache"


def parse_args() -> argparse.Namespace:
    '''解析命令行参数。

    这个函数负责接收用户从命令行传入的图片路径、模型名称和候选类别。
    这样脚本就不是写死的，可以很方便地复用到别的图片和别的标签集合上。
    '''
    parser = argparse.ArgumentParser(
        description="Use OpenAI CLIP for zero-shot image classification."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=SCRIPT_DIR / "dog.jpg",
        help="Path to the image file. Default: dog.jpg",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"CLIP model name. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=DEFAULT_LABELS,
        help="Candidate labels. Example: --labels dog cat horse airplane",
    )
    return parser.parse_args()


def build_prompts(labels: list[str]) -> list[str]:
    '''把类别标签改写成 CLIP 更容易理解的自然语言提示词。

    CLIP 的核心是比较“图片”和“文本描述”的相似度，
    所以这里不会直接使用 dog、cat 这样的裸标签，
    而是改写成 a photo of a dog 这种句子形式。
    '''
    # Zero-shot classification works by turning labels into natural-language prompts.
    return [f"a photo of a {label}" for label in labels]


def main() -> None:
    '''执行 CLIP zero-shot 图像分类的完整流程。

    整体链路如下：
    1. 读取命令行参数和本地图片；
    2. 加载 CLIP 模型与图像预处理流程；
    3. 把候选类别转成自然语言提示词；
    4. 分别编码图片和文本；
    5. 计算图文相似度并转成概率；
    6. 输出每个候选类别的概率以及最终预测结果。
    '''
    args = parse_args()
    image_path = args.image.resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading model: {args.model}")

    # Load the CLIP image encoder + text encoder and the matching preprocess pipeline.
    model, preprocess = clip.load(
        args.model,
        device=device,
        download_root=str(MODEL_CACHE_DIR),
    )
    model.eval()

    # Convert the local image into the tensor format expected by CLIP.
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    prompts = build_prompts(args.labels)
    # Convert candidate text prompts into token ids.
    text = clip.tokenize(prompts).to(device)

    with torch.no_grad():
        # CLIP returns image-text similarity scores, not class ids directly.
        logits_per_image, _ = model(image, text)
        # Softmax converts similarity scores into normalized probabilities.
        probabilities = logits_per_image.softmax(dim=-1)[0].cpu()

    best_index = int(torch.argmax(probabilities).item())

    print(f"\nImage: {image_path}")
    print("Candidate labels and probabilities:")
    for label, prompt, probability in zip(args.labels, prompts, probabilities):
        print(f"  {label:<10} | {prompt:<24} | {probability.item():.4f}")

    print("\nTop-1 prediction:")
    print(f"  {args.labels[best_index]} ({probabilities[best_index].item():.4f})")


if __name__ == "__main__":
    main()
