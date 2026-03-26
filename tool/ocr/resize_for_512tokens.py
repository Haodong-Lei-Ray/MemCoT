#!/usr/bin/env python3
"""
将图片缩放至约 512 tokens 的分辨率（针对 patch-based 模型，如 gpt-4.1-mini）。

Token 估算（patch-based，32×32 patches，multiplier 1.62）：
  token ≈ ceil(W/32) * ceil(H/32) * 1.62
  512 tokens → ~316 patches → 约 576×576（18×18=324 patches → ~525 tokens）

注：gpt-4o-mini (tile-based) 单图最少 ~2.8k tokens，无法达到 512。
    gpt-4o (tile-based) 最小约 4 tiles = 765 tokens。

Usage:
  python resize_for_512tokens.py -i /path/to/image.png
  python resize_for_512tokens.py -i in.png -o out.png --max-side 576
"""

import argparse
from pathlib import Path

from PIL import Image


def resize_for_target_tokens(
    img_path: str | Path,
    out_path: str | Path | None = None,
    target_tokens: int = 512,
    multiplier: float = 1.62,
) -> Path:
    """
    缩放图片使 patch-based 模型消耗约 target_tokens。

    target_tokens ≈ patches * multiplier  →  patches ≈ target_tokens / multiplier
    保持宽高比，长边不超过 max_side。
    """
    img_path = Path(img_path)
    if out_path is None:
        out_path = img_path.parent / f"{img_path.stem}_512tok{img_path.suffix}"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    patches_target = target_tokens / multiplier  # ~316 for 512 tokens
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    ratio = min(w, h) / max(w, h)  # <1 for non-square
    # 非正方形需更大边长：max_side ≈ 32 * sqrt(patches_target / ratio)
    max_side = int(32 * (patches_target / max(ratio, 0.3)) ** 0.5)

    if max(w, h) <= max_side:
        # 已足够小，直接保存
        img.save(out_path, quality=95)
        print(f"Image already small enough ({w}×{h}), saved to {out_path}")
        return out_path

    # 按比例缩放，长边 = max_side
    scale = max_side / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    # 对齐到 32 的倍数（向上取整以接近目标 token 数）
    new_w = max(32, ((new_w + 31) // 32) * 32)
    new_h = max(32, ((new_h + 31) // 32) * 32)

    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    resized.save(out_path, quality=95)

    patches = (new_w // 32) * (new_h // 32)
    est_tokens = int(patches * multiplier)
    print(f"Resized {w}×{h} → {new_w}×{new_h}")
    print(f"Estimated patches: {patches}, tokens: ~{est_tokens}")
    print(f"Saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Resize image for ~512 token usage")
    parser.add_argument("-i", "--input", required=True, help="Input image path")
    parser.add_argument("-o", "--output", default=None, help="Output path (default: input_512tok.ext)")
    parser.add_argument("--target-tokens", type=int, default=512, help="Target token count (default: 512)")
    args = parser.parse_args()

    resize_for_target_tokens(
        args.input,
        args.output,
        target_tokens=args.target_tokens,
    )


if __name__ == "__main__":
    main()
