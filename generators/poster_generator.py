from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "posters"


def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_font(
    size: int, bold: bool = False
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simkai.ttf",
        r"C:\Windows\Fonts\arialuni.ttf",
    ]
    if bold:
        candidates = [
            candidates[1],
            candidates[0],
            candidates[2],
            candidates[3],
            candidates[4],
        ]

    for candidate in candidates:
        font_path = Path(candidate)
        if font_path.exists():
            try:
                return ImageFont.truetype(str(font_path), size=size)
            except Exception:
                continue

    return ImageFont.load_default()


def _draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    position: tuple[int, int],
    font,
    fill,
    max_width: int,
    line_spacing: int = 10,
) -> int:
    x, y = position
    paragraphs = text.split("\n")
    for paragraph in paragraphs:
        words = paragraph.split(" ") if " " in paragraph else list(paragraph)
        joiner = " " if " " in paragraph else ""
        current_line = ""
        for word in words:
            candidate = f"{current_line}{joiner}{word}".strip()
            bbox = draw.textbbox((0, 0), candidate, font=font)
            width = bbox[2] - bbox[0]
            if width <= max_width:
                current_line = candidate
            else:
                if current_line:
                    draw.text((x, y), current_line, font=font, fill=fill)
                    y += font.size + line_spacing if hasattr(font, "size") else 24
                current_line = word
        if current_line:
            draw.text((x, y), current_line, font=font, fill=fill)
            y += font.size + line_spacing if hasattr(font, "size") else 24
        y += line_spacing
    return y


def _text_size(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont
) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _draw_pill(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
    text_fill: str,
) -> None:
    x, y = xy
    text_width, text_height = _text_size(draw, text, font)
    padding_x = 22
    padding_y = 10
    draw.rounded_rectangle(
        (
            x,
            y,
            x + text_width + padding_x * 2,
            y + text_height + padding_y * 2,
        ),
        radius=24,
        fill=fill,
    )
    draw.text((x + padding_x, y + padding_y - 2), text, font=font, fill=text_fill)


def _draw_card_shadow(
    draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], radius: int
) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(
        (left + 8, top + 10, right + 8, bottom + 10),
        radius=radius,
        fill="#eadfd5",
    )


def generate_security_poster() -> str:
    _ensure_output_dir()
    path = OUTPUT_DIR / f"phishing_security_poster_{_timestamp()}.png"

    width, height = 1200, 1600
    image = Image.new("RGB", (width, height), "#f7f3ec")
    draw = ImageDraw.Draw(image)

    title_font = _load_font(66, bold=True)
    subtitle_font = _load_font(30)
    card_title_font = _load_font(34, bold=True)
    body_font = _load_font(29)
    number_font = _load_font(30, bold=True)
    small_font = _load_font(24)
    pill_font = _load_font(24, bold=True)

    draw.rectangle((0, 0, width, 320), fill="#8f171b")
    draw.rectangle((0, 300, width, 320), fill="#c9a063")
    draw.polygon(((0, 320), (1200, 320), (1200, 405), (0, 345)), fill="#efe6d9")

    draw.rounded_rectangle((80, 58, 168, 146), radius=24, fill="#f4c95d")
    draw.line((105, 92, 125, 112, 148, 80), fill="#8f171b", width=9)
    draw.ellipse((97, 84, 114, 101), fill="#8f171b")

    _draw_pill(draw, (82, 182), "防钓鱼安全提醒", pill_font, "#f4c95d", "#4d1f17")
    _draw_pill(draw, (334, 182), "账号保护", pill_font, "#ffffff", "#8f171b")

    draw.text((190, 64), "警惕钓鱼网站", font=title_font, fill="#ffffff")
    draw.text(
        (192, 155),
        "识别可疑链接，保护账号与密码安全",
        font=subtitle_font,
        fill="#f7e7dd",
    )

    tips = [
        ("敏感信息", "不要在来源不明的页面输入账号、密码、验证码。"),
        ("域名核对", "访问前核对官网域名，警惕拼写相似和多级子域名。"),
        ("链接来源", "谨慎点击陌生链接，尤其是短信、邮件和社交消息中的链接。"),
        ("连接安全", "访问敏感网站前确认使用 HTTPS，并检查证书信息。"),
        ("高风险处理", "遇到高风险链接应立即关闭页面，并重新通过官方入口访问。"),
    ]

    intro = "遇到要求登录、转账、领取福利或填写验证码的链接时，先停下来核对来源。"
    draw.rounded_rectangle((80, 370, 1120, 455), radius=22, fill="#fffdf8")
    _draw_wrapped_text(draw, intro, (120, 392), body_font, "#4a3730", 1000, line_spacing=6)

    box_y = 500
    box_height = 156
    for index, (heading, content) in enumerate(tips):
        top = box_y + index * (box_height + 24)
        card_box = (80, top, 1120, top + box_height)
        _draw_card_shadow(draw, card_box, 24)
        draw.rounded_rectangle(
            card_box,
            radius=24,
            fill="#fffdf8",
            outline="#ead4c4",
            width=2,
        )
        draw.rounded_rectangle(
            (112, top + 42, 176, top + 106),
            radius=18,
            fill="#8f171b",
        )
        number = f"{index + 1:02d}"
        number_width, number_height = _text_size(draw, number, number_font)
        draw.text(
            (112 + (64 - number_width) / 2, top + 42 + (64 - number_height) / 2 - 3),
            number,
            font=number_font,
            fill="#ffffff",
        )
        draw.text((215, top + 34), heading, font=card_title_font, fill="#241b18")
        _draw_wrapped_text(
            draw,
            content,
            (215, top + 86),
            body_font,
            "#5a4640",
            820,
            line_spacing=6,
        )

    footer_text = "安全提示图由机器学习模型 + LangGraph 工作流自动生成"
    draw.line((80, 1445, 1120, 1445), fill="#d8c6b4", width=2)
    draw.text((80, 1486), footer_text, font=small_font, fill="#6b5750")
    draw.text(
        (80, 1528),
        datetime.now().strftime("生成时间：%Y-%m-%d %H:%M:%S"),
        font=small_font,
        fill="#6b5750",
    )

    image.save(path)
    return str(path)
