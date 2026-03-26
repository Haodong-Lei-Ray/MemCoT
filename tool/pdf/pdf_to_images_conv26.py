import argparse
import os
import re
import fitz  # PyMuPDF

# 路径由 --conv-id 或显式参数指定，main() 中设置
PDF_DIR = None
OUT_BASE = None

# 输出分辨率控制：target_height=512 时按高度 512 等比缩放；zoom 生效当 target_height 为 None
TARGET_HEIGHT = 512 + 128  # 固定高度，比例不变；设为 None 则用 zoom
ZOOM = 2.0  # 仅当 TARGET_HEIGHT 为 None 时生效

def pdf_to_images(pdf_path, out_dir, zoom=None, target_height=None):
    """
    把一个 PDF 的每一页转成 PNG 图片。
    target_height: 输出高度（px），保持比例；为 None 时用 zoom
    zoom: 缩放倍数，target_height 为 None 时生效
    """
    zoom = zoom if zoom is not None else ZOOM
    target_height = target_height if target_height is not None else TARGET_HEIGHT

    doc = fitz.open(pdf_path)
    base = os.path.splitext(os.path.basename(pdf_path))[0]

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        rect = page.rect
        if target_height is not None:
            scale = target_height / rect.height
        else:
            scale = zoom
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat)
        out_name = f"{base}_page{page_idx + 1}.png"
        out_path = os.path.join(out_dir, out_name)
        pix.save(out_path)
        print(f"Saved: {out_path}")

    doc.close()

def parse_args():
    parser = argparse.ArgumentParser(description="将 PDF 转为按 session 分组的 PNG 图片")
    parser.add_argument("--conv-id", type=str, help="conv id，如 conv-26，用于推导路径")
    parser.add_argument("--data-root", type=str, default="/mnt/petrelfs/leihaodong/ICML/locomo/data",
                        help="数据根目录")
    parser.add_argument("--pdf-dir", type=str, help="覆盖：PDF 所在目录")
    parser.add_argument("--output-dir", type=str, help="覆盖：图片输出根目录（下建 D1、D2...）")
    parser.add_argument("--target-height", type=int, default=None,
                        help="输出高度（px），等比缩放；不设则用 --zoom")
    parser.add_argument("--zoom", type=float, default=2.0, help="缩放倍数，target-height 未设时生效")
    args = parser.parse_args()

    global PDF_DIR, OUT_BASE, TARGET_HEIGHT, ZOOM
    data_root = os.path.abspath(args.data_root)
    if args.conv_id:
        # 与 json_conv26_to_pdf.py 默认输出目录对齐：data/pdf_minor/<conv-id>/
        PDF_DIR = args.pdf_dir or os.path.join(data_root, "pdf_minor", args.conv_id)
        OUT_BASE = args.output_dir or os.path.join(data_root, "img_pdf_con", args.conv_id)
    else:
        if not args.pdf_dir or not args.output_dir:
            parser.error("未指定 --conv-id 时，必须同时指定 --pdf-dir --output-dir")
        PDF_DIR = args.pdf_dir
        OUT_BASE = args.output_dir
    if args.target_height is not None:
        TARGET_HEIGHT = args.target_height
    ZOOM = args.zoom
    return args


def main():
    parse_args()
    os.makedirs(OUT_BASE, exist_ok=True)
    pdf_files = []
    for root, _, files in os.walk(PDF_DIR):
        for name in sorted(files):
            if name.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, name))
    for pdf_path in pdf_files:
        # conv-26_session_5.pdf -> D5
        m = re.search(r"session_(\d+)", os.path.basename(pdf_path), re.I)
        session_num = m.group(1) if m else "0"
        session_dir = os.path.join(OUT_BASE, f"D{session_num}")
        os.makedirs(session_dir, exist_ok=True)
        print(f"Processing {pdf_path} -> {session_dir} ...")
        pdf_to_images(pdf_path, session_dir)

if __name__ == "__main__":
    main()