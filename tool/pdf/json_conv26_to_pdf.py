import argparse
import json
import os
import re
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# 路径由 --conv-id 或显式参数指定，main() 中设置
JSON_PATH = None
IMG_DIR = None
OUTPUT_DIR = None
CONV_ID = None  # 用于生成 PDF 文件名，如 conv-26_session_1.pdf
SKIP_EXISTING = False

# 页边距（pt），0=无边距
bian = 0
LEFT_MARGIN = bian
RIGHT_MARGIN = bian
TOP_MARGIN = 0
BOTTOM_MARGIN = bian

# 标题最小上边距（pt），边距为0时避免标题被页顶裁切
TITLE_TOP_INSET = 12

PAGE_WIDTH, PAGE_HEIGHT = A4
LINE_HEIGHT = 8  # 段落间行距
WRAP_LINE_SPACING = 11  # 同一对话内换行时的行间距（pt），长句折行后的行距
TITLE_FONT_SIZE = 12  # session 标题字号
BODY_FONT_SIZE = 12  # 正文字号

# 图片高度限制：0.3=可用高度30%，None=不限制
IMG_MAX_HEIGHT_FRACTION = 0.2
IMG_MIN_HEADROOM = 80  # 换页预留空间（pt）
TURN_SPACING = 0.6  # 每轮对话后空行比例（相对 LINE_HEIGHT）


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _effective_top():
    """有效上边距，边距为0时保证标题不被裁切"""
    return max(TOP_MARGIN, TITLE_TOP_INSET)


def new_canvas(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    c = canvas.Canvas(path, pagesize=A4)
    y_start = PAGE_HEIGHT - _effective_top()
    return c, y_start


def draw_text(c, text, x, y, max_width=None, font_size=None):
    """
    在 (x, y) 位置画文本，支持简单的按宽度换行。
    返回最终的 y 坐标（即下一行起始的 y）。
    """
    if max_width is None:
        max_width = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN
    if font_size is None:
        font_size = BODY_FONT_SIZE

    from reportlab.pdfbase.pdfmetrics import stringWidth
    font_name = "Helvetica"
    c.setFont(font_name, font_size)

    words = text.split()
    line = ""
    for w in words:
        candidate = (line + " " + w).strip()
        if stringWidth(candidate, font_name, font_size) <= max_width:
            line = candidate
        else:
            c.drawString(x, y, line)
            y -= WRAP_LINE_SPACING
            line = w
    if line:
        c.drawString(x, y, line)
        y -= WRAP_LINE_SPACING
    return y


def draw_image(c, img_path, x, y, max_width=None):
    """
    在 (x, y) 下方画图像（top-left 对齐），自动缩放宽度不超过 max_width，
    高度自适应。若 IMG_MAX_HEIGHT_FRACTION 已设置，则高度不超过该比例的可视区域。
    返回画完后新的 y（图像下方一行的 y）。
    """
    if max_width is None:
        max_width = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN

    img = ImageReader(img_path)
    iw, ih = img.getSize()

    scale = min(max_width / iw, 1.0)  # 宽度不超过 max_width，不放大
    if IMG_MAX_HEIGHT_FRACTION is not None:
        available_height = PAGE_HEIGHT - _effective_top() - BOTTOM_MARGIN
        max_img_height = available_height * IMG_MAX_HEIGHT_FRACTION
        scale = min(scale, max_img_height / ih)  # 高度不超过可用高度的一半
    dw = iw * scale
    dh = ih * scale

    # 如果图片高度超出当前页剩余空间，就换页
    if y - dh < BOTTOM_MARGIN:
        c.showPage()
        y = PAGE_HEIGHT - _effective_top()

    # reportlab 的坐标原点在左下角，因此图片左下角 y 坐标为 y - dh
    c.drawImage(img, x, y - dh, width=dw, height=dh)
    y = y - dh - LINE_HEIGHT  # 图下方再空一行
    return y


def normalize_dia_id_to_filename(dia_id):
    """
    dia_id 形如 'D1:5' -> 'D1_5.jpg' 或 'D1_5.png'
    """
    # 安全起见，用正则取出 D + 数字 : 数字
    m = re.match(r"(D\d+):(\d+)", dia_id)
    if not m:
        return None

    base = f"{m.group(1)}_{m.group(2)}"
    # 先试 jpg，再试 png
    for ext in [".jpg", ".jpeg", ".png"]:
        candidate = os.path.join(IMG_DIR, base + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def generate_pdf_for_session(session_idx, session_date_time, dialogs):
    """
    为一个 session 生成 PDF。
    session_idx: int，比如 1、2
    session_date_time: 字符串，如 '1:56 pm on 8 May, 2023'
    dialogs: session_i 列表
    """
    filename = f"{CONV_ID}_session_{session_idx}.pdf"
    out_path = os.path.join(OUTPUT_DIR, filename)
    print(out_path)
    if SKIP_EXISTING and os.path.exists(out_path):
        print(f"Skip (已存在): {filename}")
        return
    c, y = new_canvas(filename)

    # 第一行写 date_time
    c.setFont("Helvetica-Bold", TITLE_FONT_SIZE)
    c.drawString(LEFT_MARGIN, y, session_date_time)
    y -= LINE_HEIGHT * 1.5

    for turn in dialogs:
        speaker = turn.get("speaker", "")
        dia_id = turn.get("dia_id", "")
        text = turn.get("text", "")

        # 1) 先写对话行：{dia_id}- {speaker}: {text}
        line_text = f"{dia_id}- {speaker}: {text}"
        if y < BOTTOM_MARGIN + LINE_HEIGHT * 2:
            c.showPage()
            y = PAGE_HEIGHT - _effective_top()
        y = draw_text(c, line_text, LEFT_MARGIN, y)

        # 2) 如果有图像信息，尝试插图像；找不到就用 blip_caption
        img_urls = turn.get("img_url") or turn.get("img_urls")
        if img_urls:
            img_path = normalize_dia_id_to_filename(dia_id)
            if img_path and os.path.exists(img_path):
                if y < BOTTOM_MARGIN + IMG_MIN_HEADROOM:
                    c.showPage()
                    y = PAGE_HEIGHT - _effective_top()
                y = draw_image(c, img_path, LEFT_MARGIN, y)
            else:
                blip_caption = turn.get("blip_caption", "")
                if blip_caption:
                    cap_text = f"[图片描述] {blip_caption}"
                    if y < BOTTOM_MARGIN + LINE_HEIGHT * 2:
                        c.showPage()
                        y = PAGE_HEIGHT - _effective_top()
                    y = draw_text(c, cap_text, LEFT_MARGIN, y)

        # 每轮对话后间距
        y -= LINE_HEIGHT * TURN_SPACING
        if y < BOTTOM_MARGIN + LINE_HEIGHT * 2:
            c.showPage()
            y = PAGE_HEIGHT - _effective_top()

    c.save()
    print(f"Saved: {filename}")


def parse_args():
    parser = argparse.ArgumentParser(description="将对话 JSON 转为 PDF")
    parser.add_argument("--conv-id", type=str, help="conv id，如 conv-26，用于推导路径")
    parser.add_argument("--data-root", type=str, default="/mnt/petrelfs/leihaodong/ICML/locomo/data",
                        help="数据根目录，默认 data/con、img_con/img、pdf_conv 等在其下")
    parser.add_argument("--json-path", type=str, help="覆盖：JSON 文件路径")
    parser.add_argument("--img-dir", type=str, help="覆盖：图片目录（按 dia_id 如 D1_5.jpg 查找）")
    parser.add_argument("--output-dir", type=str, help="覆盖：PDF 输出目录")
    parser.add_argument("--skip-existing", action="store_true",
                        help="若输出目录中已有该 session 的 PDF 则跳过，不重新生成")
    args = parser.parse_args()

    global JSON_PATH, IMG_DIR, OUTPUT_DIR, CONV_ID, SKIP_EXISTING
    data_root = os.path.abspath(args.data_root)
    if args.conv_id:
        CONV_ID = args.conv_id
        JSON_PATH = args.json_path or os.path.join(data_root, "con", f"{args.conv_id}.json")
        IMG_DIR = args.img_dir or os.path.join(data_root, "img_con", "img", args.conv_id)
        OUTPUT_DIR = args.output_dir or os.path.join(data_root, "pdf_minor", args.conv_id)
    else:
        if not args.json_path or not args.img_dir or not args.output_dir:
            parser.error("未指定 --conv-id 时，必须同时指定 --json-path --img-dir --output-dir")
        JSON_PATH = args.json_path
        IMG_DIR = args.img_dir
        OUTPUT_DIR = args.output_dir
        CONV_ID = os.path.splitext(os.path.basename(JSON_PATH))[0]  # 从 json 文件名推断
    SKIP_EXISTING = args.skip_existing
    return args


def main():
    parse_args()
    print(f"JSON_PATH: {JSON_PATH}")
    print(f"IMG_DIR: {IMG_DIR}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")

    ensure_output_dir()
    print("输出目录已创建/存在")

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"顶层 keys: {list(data.keys())[:10]}")

    # 找出所有 session_i
    # 例如 session_1_date_time / session_1, session_2_date_time / session_2 ...
    session_idx = 1
    while True:
        date_key = f"session_{session_idx}_date_time"
        sess_key = f"session_{session_idx}"
        print(f"检查 {date_key}, {sess_key}")

        if date_key not in data or sess_key not in data:
            print("找不到更多 session，结束。")
            break

        session_date_time = data[date_key]
        dialogs = data[sess_key]
        print(f"生成 session {session_idx}, 对话轮数: {len(dialogs)}")

        generate_pdf_for_session(session_idx, session_date_time, dialogs)
        session_idx += 1


if __name__ == "__main__":
    main()