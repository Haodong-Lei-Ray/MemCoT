import json
import os
import re
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

JSON_PATH = "/mnt/petrelfs/leihaodong/ICML/locomo/data/con/conv-26.json"
IMG_DIR = "/mnt/petrelfs/leihaodong/ICML/locomo/data/img_con/img/conv-26"
OUTPUT_DIR = "/mnt/petrelfs/leihaodong/ICML/locomo/data/pdf_conv26"

PAGE_WIDTH, PAGE_HEIGHT = A4
LEFT_MARGIN = 50
RIGHT_MARGIN = 50
TOP_MARGIN = 50
BOTTOM_MARGIN = 50
LINE_HEIGHT = 16  # 行距


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def new_canvas(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    c = canvas.Canvas(path, pagesize=A4)
    # 从上往下画，先设置当前 y
    y_start = PAGE_HEIGHT - TOP_MARGIN
    return c, y_start


def draw_text(c, text, x, y, max_width=None):
    """
    在 (x, y) 位置画文本，支持简单的按宽度换行。
    返回最终的 y 坐标（即下一行起始的 y）。
    """
    if max_width is None:
        max_width = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN

    # reportlab 字体宽度估计：用 stringWidth
    from reportlab.pdfbase.pdfmetrics import stringWidth
    font_name = "Helvetica"
    font_size = 12
    c.setFont(font_name, font_size)

    words = text.split()
    line = ""
    for w in words:
        candidate = (line + " " + w).strip()
        if stringWidth(candidate, font_name, font_size) <= max_width:
            line = candidate
        else:
            c.drawString(x, y, line)
            y -= LINE_HEIGHT
            line = w
    if line:
        c.drawString(x, y, line)
        y -= LINE_HEIGHT
    return y


def draw_image(c, img_path, x, y, max_width=None):
    """
    在 (x, y) 下方画图像（top-left 对齐），自动缩放宽度不超过 max_width，
    高度自适应。返回画完后新的 y（图像下方一行的 y）。
    """
    if max_width is None:
        max_width = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN

    img = ImageReader(img_path)
    iw, ih = img.getSize()

    scale = min(max_width / iw, 1.0)  # 宽度不超过 max_width，不放大
    dw = iw * scale
    dh = ih * scale

    # 如果图片高度超出当前页剩余空间，就换页
    if y - dh < BOTTOM_MARGIN:
        c.showPage()
        y = PAGE_HEIGHT - TOP_MARGIN

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
    filename = f"conv-26_session_{session_idx}.pdf"
    c, y = new_canvas(filename)

    # 第一行写 date_time
    c.setFont("Helvetica-Bold", 14)
    c.drawString(LEFT_MARGIN, y, session_date_time)
    y -= LINE_HEIGHT * 2

    for turn in dialogs:
        speaker = turn.get("speaker", "")
        dia_id = turn.get("dia_id", "")
        text = turn.get("text", "")

        # 1) 先写对话行：{dia_id}- {speaker}: {text}
        line_text = f"{dia_id}- {speaker}: {text}"
        if y < BOTTOM_MARGIN + LINE_HEIGHT * 3:
            c.showPage()
            y = PAGE_HEIGHT - TOP_MARGIN
        y = draw_text(c, line_text, LEFT_MARGIN, y)

        # 2) 如果有图像信息，尝试插图像；找不到就用 blip_caption
        img_urls = turn.get("img_url") or turn.get("img_urls")  # 以防 key 名不同
        if img_urls:
            # 用 dia_id 去本地目录里找图片
            img_path = normalize_dia_id_to_filename(dia_id)
            if img_path and os.path.exists(img_path):
                # 插入图片
                if y < BOTTOM_MARGIN + 150:  # 简单预留空间
                    c.showPage()
                    y = PAGE_HEIGHT - TOP_MARGIN
                y = draw_image(c, img_path, LEFT_MARGIN, y)
            else:
                # 找不到图，则用 blip_caption 文字
                blip_caption = turn.get("blip_caption", "")
                if blip_caption:
                    cap_text = f"[图片描述] {blip_caption}"
                    if y < BOTTOM_MARGIN + LINE_HEIGHT * 3:
                        c.showPage()
                        y = PAGE_HEIGHT - TOP_MARGIN
                    y = draw_text(c, cap_text, LEFT_MARGIN, y)

        # 每个 turn 之间空一行
        y -= LINE_HEIGHT / 2
        if y < BOTTOM_MARGIN + LINE_HEIGHT * 3:
            c.showPage()
            y = PAGE_HEIGHT - TOP_MARGIN

    c.save()
    print(f"Saved: {filename}")


def main():
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