#!/usr/bin/env python3
"""
将 conv-26-high 内所有图片批量缩放到统一分辨率，使 patch-based 模型消耗约 384 tokens。
输出到 conv-26，保持 D1、D2... 子目录结构。
所有输出图片尺寸一致（固定 416×448 ≈ 182 patches → ~295 tokens，或 448×448 ≈ 196 → ~317；
为接近 384：240 patches → 480×512）。
"""

import argparse
from pathlib import Path

from PIL import Image

# 384 tokens → 384/1.62 ≈ 237 patches. 15×16=240 → 480×512
TARGET_TOKENS = 384
MULTIPLIER = 1.62
# 固定输出尺寸（所有图片统一）
OUT_W = 480
OUT_H = 512

INPUT_DIR = Path("/mnt/petrelfs/leihaodong/ICML/locomo/data/img_pdf_con/conv-26-high")
OUTPUT_DIR = Path("/mnt/petrelfs/leihaodong/ICML/locomo/data/img_pdf_con/conv-26")
EXTENSIONS = (".png", ".jpg", ".jpeg")


def resize_to_fixed(img_path: Path, out_path: Path, target_w: int, target_h: int) -> None:
    """缩放到固定尺寸，保持宽高比并 letterbox 填充至 target_w×target_h。"""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    # 居中贴到 target_w×target_h 画布
    canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))
    canvas.save(out_path, quality=95)


def main():
    parser = argparse.ArgumentParser(description="Batch resize images to ~384 tokens, uniform size")
    parser.add_argument("-i", "--input-dir", default=str(INPUT_DIR), help="Input directory")
    parser.add_argument("-o", "--output-dir", default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--target-tokens", type=int, default=TARGET_TOKENS)
    parser.add_argument("--out-w", type=int, default=OUT_W)
    parser.add_argument("--out-h", type=int, default=OUT_H)
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_w, out_h = args.out_w, args.out_h

    patches = (out_w // 32) * (out_h // 32)
    est = int(patches * MULTIPLIER)
    print(f"Target: {args.target_tokens} tokens | Output size: {out_w}×{out_h} | Est. ~{est} tokens")

    count = 0
    for img_path in sorted(in_dir.rglob("*")):
        if not img_path.is_file() or img_path.suffix.lower() not in EXTENSIONS:
            continue
        rel = img_path.relative_to(in_dir)
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        resize_to_fixed(img_path, out_path, out_w, out_h)
        count += 1
        if count <= 5 or count % 50 == 0:
            print(f"  [{count}] {rel}")

    print(f"Done. {count} images saved to {out_dir}")


if __name__ == "__main__":
    main()
