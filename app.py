from collections import Counter
from datetime import datetime
from pathlib import Path
import base64
import hashlib
import io
import json

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageFont
try:
    from streamlit_cropper import st_cropper
except ImportError:
    st_cropper = None
from ultralytics import YOLO


APP_DIR = Path(__file__).resolve().parent
# ตรวจสอบการเรียกใช้โมเดลจากผลการเทรนรอบที่ 6 (train-6)
MODEL_PATH = APP_DIR / "train-6" / "weights" / "best.pt"
FEEDBACK_DIR = APP_DIR / "feedback"
FEEDBACK_FILE = FEEDBACK_DIR / "corrections.jsonl"
CONFIDENCE_THRESHOLD = 0.40
REVIEW_IMAGE_MAX_WIDTH = 800
REVIEW_IMAGE_MAX_HEIGHT = 800

CELL_PROFILES = {
    "RBC": {
        "label": "เม็ดเลือดแดง",
        "base_score": 18,
        "note": "ลักษณะที่พบสอดคล้องกับกลุ่มเม็ดเลือดแดงในภาพตัวอย่าง",
    },
    "platelet": {
        "label": "เกล็ดเลือด",
        "base_score": 36,
        "note": "พบเกล็ดเลือด ควรอ่านร่วมกับจำนวนและตำแหน่งในภาพสเมียร์",
    },
    "lymphocyte": {
        "label": "ลิมโฟไซต์",
        "base_score": 52,
        "note": "พบเม็ดเลือดขาวชนิด lymphocyte เหมาะกับการทวนผลโดยผู้เชี่ยวชาญ",
    },
    "monocyte": {
        "label": "โมโนไซต์",
        "base_score": 58,
        "note": "พบเม็ดเลือดขาวชนิด monocyte ควรตรวจทานตำแหน่งและรูปร่างซ้ำ",
    },
    "neutrophil": {
        "label": "นิวโทรฟิล",
        "base_score": 62,
        "note": "พบเม็ดเลือดขาวชนิด neutrophil ควรอ่านร่วมกับอาการและผล CBC",
    },
    "basophil": {
        "label": "เบโซฟิล",
        "base_score": 65,
        "note": "พบเม็ดเลือดขาวชนิด basophil ตรวจสอบความหนาแน่นของแกรนูลเข้ม",
    },
    "eosinophil": {
        "label": "อีโอซิโนฟิล",
        "base_score": 60,
        "note": "พบเม็ดเลือดขาวชนิด eosinophil สังเกตลักษณะแกรนูลสีส้ม/ชมพูเด่นชัด",
    },
}

BOX_COLORS = {
    "RBC": "#44e7d0",
    "platelet": "#ffd166",
    "lymphocyte": "#71a7ff",
    "monocyte": "#ff7aa2",
    "neutrophil": "#ff3d61",
    "basophil": "#b9f56a",
    "eosinophil": "#ffa444",
}


st.set_page_config(
    page_title="Abnormal Blood Cell Detection",
    page_icon="🩸",
    layout="wide",
)

# สไตล์ตกแต่ง UI (CSS Grid & Glassmorphism)
st.markdown(
    """
    <style>
    :root {
        --panel: rgba(255,255,255,.075);
        --panel-2: rgba(255,255,255,.115);
        --stroke: rgba(255,255,255,.16);
        --text: #f7fbff;
        --muted: #a9b6c7;
        --red: #ff3d61;
        --cyan: #44e7d0;
        --lime: #b9f56a;
        --amber: #ffd166;
    }

    .stApp {
        color: var(--text);
        background:
            radial-gradient(circle at 12% 5%, rgba(255, 61, 97, .24), transparent 28%),
            radial-gradient(circle at 88% 0%, rgba(68, 231, 208, .18), transparent 26%),
            linear-gradient(135deg, #080b12 0%, #101522 48%, #130912 100%);
        font-family: Inter, "Segoe UI", Tahoma, sans-serif;
    }

    .block-container {
        min-height: 100vh;
        padding: clamp(28px, 6vh, 72px) clamp(22px, 7vw, 96px) 38px;
        max-width: 100%;
        background:
            radial-gradient(circle at 78% 34%, rgba(255, 61, 97, .30), transparent 18%),
            radial-gradient(circle at 22% 72%, rgba(68, 231, 208, .18), transparent 24%),
            linear-gradient(120deg, rgba(255,61,97,.22), rgba(68,231,208,.12)),
            repeating-linear-gradient(90deg, rgba(255,255,255,.035) 0 1px, transparent 1px 34px);
        box-shadow: 0 24px 70px rgba(0,0,0,.35);
        position: relative;
        overflow-x: hidden;
    }

    .hero {
        min-height: 100vh;
        border: 1px solid var(--stroke);
        border-left: 0;
        border-right: 0;
        border-radius: 0;
        padding: clamp(48px, 9vh, 110px) clamp(24px, 7vw, 96px);
        background:
            radial-gradient(circle at 78% 34%, rgba(255, 61, 97, .30), transparent 18%),
            radial-gradient(circle at 22% 72%, rgba(68, 231, 208, .18), transparent 24%),
            linear-gradient(120deg, rgba(255,61,97,.22), rgba(68,231,208,.12)),
            repeating-linear-gradient(90deg, rgba(255,255,255,.035) 0 1px, transparent 1px 34px);
        box-shadow: 0 24px 70px rgba(0,0,0,.35);
        position: relative;
        overflow: hidden;
        margin: 0 calc(50% - 50vw) 26px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .hero.scan-mode {
        padding-top: clamp(28px, 5vh, 54px);
        justify-content: flex-start;
    }

    .hero:after {
        content: "";
        position: absolute;
        inset: auto 7vw 9vh auto;
        width: clamp(180px, 22vw, 320px);
        height: clamp(180px, 22vw, 320px);
        border-radius: 50%;
        background:
            radial-gradient(circle at 48% 46%, rgba(255,255,255,.92) 0 10%, transparent 11%),
            radial-gradient(circle, rgba(255,61,97,.95) 0 58%, rgba(126,14,42,.92) 59% 100%);
        box-shadow: 0 0 42px rgba(255,61,97,.35);
        opacity: .92;
    }

    .blood-cell {
        position: absolute;
        border-radius: 50%;
        pointer-events: none;
        opacity: .72;
        background:
            radial-gradient(circle at 42% 40%, rgba(255,255,255,.85) 0 12%, transparent 13%),
            radial-gradient(circle at 50% 50%, rgba(255,94,119,.98) 0 58%, rgba(129,14,43,.95) 60% 100%);
        box-shadow: 0 0 28px rgba(255, 61, 97, .28);
        animation: float-cell 9s ease-in-out infinite;
        z-index: 0;
    }

    .cell-a { width: 92px; height: 92px; left: 68%; top: 18%; animation-delay: -1s; }
    .cell-b { width: 58px; height: 58px; left: 84%; top: 52%; animation-delay: -4s; opacity: .55; }
    .cell-c { width: 70px; height: 70px; left: 54%; top: 70%; animation-delay: -6s; opacity: .48; }
    .cell-d { width: 42px; height: 42px; left: 91%; top: 24%; animation-delay: -2.5s; opacity: .42; }
    .cell-e { width: 50px; height: 50px; left: 12%; top: 14%; animation-delay: -3.5s; opacity: .4; }
    .cell-f { width: 80px; height: 80px; left: 6%; top: 56%; animation-delay: -7s; opacity: .5; }
    .cell-g { width: 36px; height: 36px; left: 32%; top: 82%; animation-delay: -5.5s; opacity: .38; }
    .cell-h { width: 64px; height: 64px; left: 44%; top: 8%; animation-delay: -8s; opacity: .45; }

    @keyframes float-cell {
        0%, 100% { transform: translate3d(0, 0, 0) rotate(0deg) scale(1); }
        30% { transform: translate3d(-18px, 16px, 0) rotate(11deg) scale(1.05); }
        58% { transform: translate3d(14px, -12px, 0) rotate(-8deg) scale(.96); }
        78% { transform: translate3d(-8px, -22px, 0) rotate(7deg) scale(1.02); }
    }

    .eyebrow {
        color: var(--cyan);
        font-weight: 850;
        letter-spacing: .08em;
        text-transform: uppercase;
        font-size: .78rem;
    }

    .title {
        margin: 10px auto 8px;
        font-size: clamp(3rem, 8vw, 7rem);
        line-height: .95;
        font-weight: 950;
        max-width: 980px;
        text-align: center;
    }

    .hero-copy {
        position: relative;
        z-index: 2;
        min-height: 58vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        max-width: 980px;
        margin: 0 auto;
        text-align: center;
    }

    .subtitle {
        max-width: 760px;
        color: var(--muted);
        font-size: 1.03rem;
        line-height: 1.65;
    }

    .trust-row {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 10px;
        margin-top: 18px;
    }

    .chip {
        border: 1px solid var(--stroke);
        border-radius: 999px;
        padding: 8px 12px;
        background: rgba(255,255,255,.08);
        color: #e8f1ff;
        font-size: .86rem;
        font-weight: 750;
    }

    .glass {
        border: 1px solid var(--stroke);
        border-radius: 8px;
        background: var(--panel);
        padding: 20px;
        box-shadow: 0 18px 52px rgba(0,0,0,.25);
        backdrop-filter: blur(12px);
        margin-bottom: 18px;
    }

    .section-title {
        font-size: 1.1rem;
        font-weight: 900;
        margin: 0 0 14px;
    }

    .risk-banner {
        border-radius: 8px;
        padding: 18px;
        border: 1px solid var(--stroke);
        background: var(--panel-2);
        margin-bottom: 16px;
    }

    .risk-low { border-left: 6px solid var(--lime); }
    .risk-medium { border-left: 6px solid var(--amber); }
    .risk-high { border-left: 6px solid var(--red); }

    .risk-title {
        font-weight: 950;
        font-size: 1.32rem;
        margin-bottom: 4px;
    }

    .risk-copy {
        color: var(--muted);
        line-height: 1.55;
        margin: 0;
    }

    .cell-row {
        display: grid;
        grid-template-columns: 1.1fr .8fr .8fr;
        gap: 10px;
        padding: 12px 0;
        border-bottom: 1px solid rgba(255,255,255,.1);
        align-items: center;
    }

    .cell-row:last-child { border-bottom: 0; }
    .cell-name { font-weight: 850; }
    .cell-note { color: var(--muted); font-size: .86rem; margin-top: 2px; }
    .pill {
        display: inline-block;
        width: fit-content;
        border-radius: 999px;
        padding: 6px 10px;
        background: rgba(255,255,255,.09);
        border: 1px solid rgba(255,255,255,.14);
        font-weight: 800;
        font-size: .84rem;
    }

    .stFileUploader {
        border: 1px dashed rgba(255,255,255,.28);
        background: rgba(255,255,255,.07);
        border-radius: 8px;
        padding: 14px;
    }

    div[data-testid="stMetric"] {
        border: 1px solid var(--stroke);
        border-radius: 8px;
        background: rgba(255,255,255,.075);
        padding: 14px;
    }

    div[data-testid="stMetricValue"] {
        color: var(--text);
        font-weight: 950;
    }

    .stDataFrame {
        border: 1px solid var(--stroke);
        border-radius: 8px;
        overflow: hidden;
    }

    .notice {
        color: var(--muted);
        font-size: .86rem;
        line-height: 1.55;
    }

    .upload-shell {
        width: min(760px, 60vw);
        margin: 24px 0 0;
        position: relative;
        z-index: 2;
    }

    .hero-upload {
        max-width: 540px;
    }

    .initial-upload {
        display: flex;
        justify-content: center;
    }

    .initial-upload [data-testid="stFileUploader"] {
        width: 100%;
        max-width: 540px;
        position: relative;
        z-index: 2;
        margin: 0 auto;
    }

    .scan-stage {
        width: min(1100px, 78vw);
        margin: 0 auto;
        position: relative;
        z-index: 2;
    }

    .scan-topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 10px;
        flex-wrap: wrap;
    }

    .scan-topbar-left {
        display: flex;
        align-items: center;
        gap: 10px;
        flex: 1;
    }

    /* ย่อ file uploader ให้เป็นแถบแคบๆ */
    .scan-uploader [data-testid="stFileUploader"] {
        border: 1px solid rgba(255,255,255,.22);
        border-radius: 8px;
        background: rgba(255,255,255,.06);
        padding: 0;
    }

    .scan-uploader [data-testid="stFileUploaderDropzone"] {
        padding: 6px 12px !important;
        min-height: unset !important;
        flex-direction: row !important;
        gap: 8px !important;
    }

    .scan-uploader [data-testid="stFileUploaderDropzoneInstructions"] {
        display: none !important;
    }

    .scan-uploader [data-testid="stFileUploader"] button {
        padding: 4px 14px !important;
        font-size: .82rem !important;
        height: 32px !important;
        min-height: unset !important;
    }

    .scan-uploader [data-testid="stFileUploader"] small {
        display: none !important;
    }

    .scan-uploader [data-testid="stFileUploader"] label {
        display: none !important;
    }

    .info-pill {
        background: rgba(255,255,255,.08);
        border: 1px solid rgba(255,255,255,.15);
        border-radius: 999px;
        padding: 5px 14px;
        font-size: .82rem;
        color: var(--muted);
        font-weight: 700;
        white-space: nowrap;
    }

    .scan-toolbar {
        display: flex;
        justify-content: flex-end;
        align-items: flex-start;
        gap: 12px;
        margin-bottom: 10px;
    }

    .scan-top-note {
        color: var(--muted);
        font-weight: 800;
        margin: 0;
    }

    .scan-toolbar [data-testid="stFileUploader"] {
        width: min(320px, 42vw);
        margin-left: auto;
    }

    .scan-controls {
        display: flex;
        justify-content: space-between;
        gap: 14px;
        align-items: center;
        margin: 0 0 12px;
        color: var(--muted);
        font-weight: 750;
    }

    .scan-image {
        border: 1px solid var(--stroke);
        border-radius: 8px;
        overflow: hidden;
        background: rgba(255,255,255,.075);
        box-shadow: 0 24px 70px rgba(0,0,0,.35);
        width: 100%;
        display: block;
    }

    .scan-image img {
        width: 100%;
        height: auto;
        display: block;
        border-radius: 8px;
    }

    .control-row {
        display: flex;
        justify-content: center;
        gap: 16px;
        align-items: center;
        flex-wrap: wrap;
        margin: 6px auto 18px;
        width: min(760px, 60vw);
    }

    .image-note {
        color: var(--muted);
        text-align: center;
        margin: 8px 0 16px;
        font-size: .92rem;
    }

    @media (max-width: 900px) {
        .upload-shell,
        .control-row,
        .scan-stage {
            width: 100%;
        }

        .scan-toolbar {
            justify-content: stretch;
        }

        .scan-toolbar [data-testid="stFileUploader"] {
            width: 100%;
        }

        .cell-a { left: 74%; top: 34px; }
        .cell-b { left: 82%; top: 152px; }
        .cell-c, .cell-d, .cell-e, .cell-f, .cell-g, .cell-h { display: none; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"ไม่พบไฟล์โมเดลในตำแหน่งที่ระบุ: {MODEL_PATH}")
        st.stop()
    return YOLO(str(MODEL_PATH))


# ปรับปรุง: ย้ายกระบวนการ Predict เข้า Cache ป้องกันสไลเดอร์หรือปุ่มรีเซ็ตทำโมเดลรันซ้ำโดยไม่จำเป็น
@st.cache_data
def predict_blood_cells(_model, img_array, conf_threshold):
    results = _model.predict(source=img_array, conf=conf_threshold, verbose=False)
    return build_detections(results[0].boxes, _model.names)


def image_fingerprint(uploaded_file) -> str:
    uploaded_file.seek(0)
    digest = hashlib.sha256(uploaded_file.getvalue()).hexdigest()[:16]
    uploaded_file.seek(0)
    return digest


def score_cell(class_name: str, confidence: float) -> tuple[int, str, str]:
    profile = CELL_PROFILES.get(
        class_name,
        {
            "label": class_name,
            "base_score": 55,
            "note": "คลาสนี้ยังไม่มีคำอธิบายเฉพาะในระบบ",
        },
    )
    confidence_penalty = max(0, int((0.70 - confidence) * 100))
    score = min(100, profile["base_score"] + confidence_penalty)

    if score >= 70:
        level = "สูง"
    elif score >= 40:
        level = "ปานกลาง"
    else:
        level = "ต่ำ"

    return score, level, profile["note"]


def overall_risk(rows: list[dict]) -> tuple[str, str, str]:
    if not rows:
        return (
            "ไม่พบเซลล์",
            "medium",
            "ไม่มีวัตถุที่ผ่านเกณฑ์หลังแก้ไข ควรตรวจคุณภาพภาพหรือลองลด threshold",
        )

    max_score = max(row["คะแนนความเสี่ยง"] for row in rows)
    review_count = sum(row["ระดับความเสี่ยง"] in {"ปานกลาง", "สูง"} for row in rows)

    if max_score >= 70 or review_count >= 3:
        return (
            "ความเสี่ยงสูง",
            "high",
            "พบรายการที่ควรตรวจทานเชิงสัณฐานวิทยาอย่างใกล้ชิด แนะนำให้ผู้เชี่ยวชาญยืนยันผล",
        )
    if max_score >= 40 or review_count > 0:
        return (
            "ความเสี่ยงปานกลาง",
            "medium",
            "มีบางเซลล์อยู่ในโหมดเฝ้าระวัง ควรอ่านร่วมกับภาพจริงและผลตรวจอื่น",
        )
    return (
        "ความเสี่ยงต่ำ",
        "low",
        "ภาพนี้มีรูปแบบที่โมเดลและคำแก้ล่าสุดประเมินว่าอยู่ในช่วงเสี่ยงต่ำ",
    )


def build_detections(boxes, names: dict[int, str]) -> list[dict]:
    detections = []
    for index, box in enumerate(boxes, start=1):
        class_id = int(box.cls[0])
        class_name = names.get(class_id, str(class_id))
        detections.append(
            {
                "id": index,
                "bbox": [round(float(value), 2) for value in box.xyxy[0].tolist()],
                "original_class": class_name,
                "corrected_class": class_name,
                "confidence": float(box.conf[0]),
                "keep": True,
            }
        )
    return detections


def apply_corrections(detections: list[dict], corrections: dict) -> list[dict]:
    corrected = []
    for detection in detections:
        key = str(detection["id"])
        current = detection.copy()
        if key in corrections:
            current["keep"] = corrections[key]["keep"]
            current["corrected_class"] = corrections[key]["corrected_class"]
        if current["keep"]:
            corrected.append(current)
    return corrected


def build_rows(detections: list[dict]) -> list[dict]:
    rows = []
    for detection in detections:
        class_name = detection["corrected_class"]
        profile = CELL_PROFILES.get(class_name, {"label": class_name})
        score, level, note = score_cell(class_name, detection["confidence"])
        rows.append(
            {
                "ID": detection["id"],
                "ชนิดเซลล์": profile.get("label", class_name),
                "คลาสโมเดลเดิม": detection["original_class"],
                "คลาสที่ยืนยัน": class_name,
                "ความมั่นใจ": round(detection["confidence"] * 100, 2),
                "คะแนนความเสี่ยง": score,
                "ระดับความเสี่ยง": level,
                "คำแนะนำ": note,
            }
        )
    return rows


def draw_boxes(image: Image.Image, detections: list[dict]) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        class_name = detection["corrected_class"]
        color = BOX_COLORS.get(class_name, "#ffffff")
        label = f"#{detection['id']} {class_name} {detection['confidence'] * 100:.1f}%"

        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label_box = draw.textbbox((x1, y1), label, font=font)
        label_w = label_box[2] - label_box[0]
        label_h = label_box[3] - label_box[1]
        top = max(0, y1 - label_h - 8)
        draw.rectangle((x1, top, x1 + label_w + 10, top + label_h + 8), fill=color)
        draw.text((x1 + 5, top + 4), label, fill="#080b12", font=font)

    return canvas


def hex_to_rgba(hex_color: str, alpha: int) -> tuple[int, int, int, int]:
    hex_color = hex_color.lstrip("#")
    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
        alpha,
    )


def remove_detected_cells(image: Image.Image, detections: list[dict]) -> Image.Image:
    if not detections:
        return image.copy()

    base = image.copy().convert("RGB")
    width, height = base.size

    # เบลอภาพทั้งใบไว้ล่วงหน้า ใช้เป็น "สีพื้นหลังที่ดูดมา" สำหรับทาทับตำแหน่งเซลล์
    blur_radius = max(6, int(min(width, height) * 0.02))
    blurred = base.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # mask แบบ soft edge (ฟุ้งขอบ) สะสมตำแหน่งเซลล์ทั้งหมดไว้ในภาพเดียว ลด artifact ตอนผสมภาพ
    mask = Image.new("L", base.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    for detection in detections:
        x1, y1, x2, y2 = [float(value) for value in detection["bbox"]]
        cell_width = max(4.0, x2 - x1)
        cell_height = max(4.0, y2 - y1)
        radius = max(cell_width, cell_height) * 0.58
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        cover_box = [cx - radius, cy - radius, cx + radius, cy + radius]
        mask_draw.ellipse(cover_box, fill=255)

    # ฟุ้งขอบ mask ให้รอยต่อระหว่างพื้นหลังเบลอกับภาพจริงดูเนียนไม่เห็นรอยตัด
    feather_radius = max(6, int(min(width, height) * 0.012))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))

    canvas = Image.composite(blurred, base, mask)
    return canvas


def load_feedback_stats() -> dict:
    if not FEEDBACK_FILE.exists():
        return {"total": 0, "changed": 0, "deleted": 0}

    stats = {"total": 0, "changed": 0, "deleted": 0}
    with FEEDBACK_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            stats["total"] += len(record.get("corrections", []))
            for item in record.get("corrections", []):
                if not item.get("keep", True):
                    stats["deleted"] += 1
                elif item.get("original_class") != item.get("corrected_class"):
                    stats["changed"] += 1
    return stats


def save_feedback(image_id: str, file_name: str, detections: list[dict], corrections: dict) -> int:
    payload_items = []

    for detection in detections:
        key = str(detection["id"])
        corrected_class = corrections.get(key, {}).get(
            "corrected_class",
            detection["corrected_class"],
        )
        keep = corrections.get(key, {}).get("keep", detection["keep"])
        was_changed = corrected_class != detection["original_class"]
        was_deleted = keep is False

        if not was_changed and not was_deleted:
            continue

        payload_items.append(
            {
                "id": detection["id"],
                "bbox_xyxy": detection["bbox"],
                "original_class": detection["original_class"],
                "corrected_class": corrected_class,
                "confidence": round(detection["confidence"], 5),
                "keep": keep,
            }
        )

    if not payload_items:
        return 0

    FEEDBACK_DIR.mkdir(exist_ok=True)
    record = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "image_id": image_id,
        "file_name": file_name,
        "model_path": str(MODEL_PATH),
        "corrections": payload_items,
    }
    with FEEDBACK_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(payload_items)


def class_display_name(class_name: str) -> str:
    profile = CELL_PROFILES.get(class_name)
    if not profile:
        return class_name
    return f"{class_name} - {profile['label']}"


def fit_review_image(image: Image.Image) -> tuple[Image.Image, float]:
    width, height = image.size
    scale = min(
        REVIEW_IMAGE_MAX_WIDTH / width,
        REVIEW_IMAGE_MAX_HEIGHT / height,
        1.0,
    )
    if scale == 1.0:
        return image.copy(), 1.0

    resized = image.resize((int(width * scale), int(height * scale)))
    return resized, scale


def on_widget_change(s_key, det_id, field, widget_key):
    st.session_state[s_key][str(det_id)][field] = st.session_state[widget_key]


def crop_box_widget(image: Image.Image, manual_class: str, image_id: str):
    if st_cropper is None:
        st.error("ยังไม่ได้ติดตั้ง streamlit-cropper ให้รัน `pip install -r requirements.txt` ก่อนใช้งานแท็บเพิ่มวงใหม่")
        return None

    cropper_args = {
        "realtime_update": False,
        "box_color": BOX_COLORS.get(manual_class, "#ffffff"),
        "aspect_ratio": None,
        "return_type": "box",
        "key": f"cropper_{image_id}_{manual_class}",
    }

    try:
        return st_cropper(
            image,
            should_resize_image=False,
            stroke_width=3,
            **cropper_args,
        )
    except TypeError:
        return st_cropper(image, **cropper_args)


# เริ่มต้นการทำงานของแอปพลิเคชัน
model = load_model()
class_options = list(CELL_PROFILES.keys())
feedback_stats = load_feedback_stats()

uploaded_file = st.session_state.get("scan_upload") or st.session_state.get("hero_upload")

st.markdown(
    """
    <span class="blood-cell cell-a"></span>
    <span class="blood-cell cell-b"></span>
    <span class="blood-cell cell-c"></span>
    <span class="blood-cell cell-d"></span>
    <span class="blood-cell cell-e"></span>
    <span class="blood-cell cell-f"></span>
    <span class="blood-cell cell-g"></span>
    <span class="blood-cell cell-h"></span>
    """,
    unsafe_allow_html=True,
)

if uploaded_file is None:
    st.markdown(
        """
        <div class="hero-copy">
            <div class="title">Abnormal Blood<br>Cell Detection</div>
            <div class="subtitle">
                ระบบ AI นี้ใช้สำหรับการตรวจวิเคราะห์และคัดกรองเบื้องต้นเท่านั้น
                ไม่สามารถใช้แทนการวินิจฉัยทางการแพทย์ได้ โปรดปรึกษาแพทย์หรือผู้เชี่ยวชาญเฉพาะทาง
            </div>
            <div class="trust-row">
                <span class="chip">อัปโหลดภาพในหน้าแรก</span>
                <span class="chip">AI สแกนอัตโนมัติ</span>
                <span class="chip">เปลี่ยนรูปได้ทันที</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="initial-upload">', unsafe_allow_html=True)
    upload_left, upload_center, upload_right = st.columns([0.22, 0.56, 0.22])
    with upload_center:
        st.file_uploader(
            "เพิ่มไฟล์เข้าเพื่อตรวจเม็ดเลือด",
            type=["jpg", "jpeg", "png"],
            key="hero_upload",
        )
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

image_id = image_fingerprint(uploaded_file)
source_img = Image.open(uploaded_file).convert("RGB")
img_array = np.array(source_img)

with st.spinner("AI กำลังตรวจหาเซลล์ในภาพ..."):
    detections = predict_blood_cells(model, img_array, CONFIDENCE_THRESHOLD)

toggle_control = getattr(st, "toggle", st.checkbox)

stage_left, stage_center, stage_right = st.columns([0.03, 0.94, 0.03])
with stage_center:
    # --- Toolbar: toggle ซ้าย | info pill กลาง | upload ขวา ---
    tb_toggle, tb_info, tb_upload = st.columns([0.38, 0.32, 0.30])

    with tb_toggle:
        hide_detected_cells = toggle_control(
            "ซ่อนเซลล์ปกติที่ AI ตรวจเจอ",
            value=True,
            help="เมื่อเปิด ระบบจะลบเซลล์ปกติออกจากภาพ เพื่อให้มองเห็นเซลล์ผิดปกติได้ชัดขึ้น",
        )

    with tb_info:
        st.markdown(
            f'<div style="padding-top:6px"><span class="info-pill">🔬 พบ {len(detections)} จุด</span></div>',
            unsafe_allow_html=True,
        )

    with tb_upload:
        st.markdown('<div class="scan-uploader">', unsafe_allow_html=True)
        st.file_uploader(
            "เพิ่มรูปใหม่",
            type=["jpg", "jpeg", "png"],
            key="scan_upload",
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    display_img = remove_detected_cells(source_img, detections) if hide_detected_cells else draw_boxes(source_img, detections)

    img_b64 = image_to_base64(display_img)
    st.markdown(
        f"""
        <div class="scan-image">
            <img src="data:image/png;base64,{img_b64}" style="width:100%;height:auto;object-fit:contain;display:block;" />
        </div>
        """,
        unsafe_allow_html=True,
    )

st.stop()

left, right = st.columns([0.92, 1.08], gap="large")

with left:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">อัปโหลดภาพสไลด์เลือด</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "เลือกไฟล์ภาพจากกล้องจุลทรรศน์",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    conf_threshold = st.slider(
        "ความมั่นใจขั้นต่ำของโมเดล",
        min_value=0.25,
        max_value=0.80,
        value=CONFIDENCE_THRESHOLD,
        step=0.05,
        help="ค่าสูงขึ้นจะลดวงที่ไม่มั่นใจ แต่บางเซลล์อาจไม่แสดง",
    )
    st.markdown(
        '<p class="notice">การกดบันทึก feedback ยังไม่เปลี่ยนน้ำหนักโมเดลทันที แต่จะเก็บเป็นบทเรียนสำหรับ train รอบต่อไป</p>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Learning Progress</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    p1.metric("บทเรียนทั้งหมด", feedback_stats["total"])
    p2.metric("แก้คลาส", feedback_stats["changed"])
    p3.metric("ลบวงผิด", feedback_stats["deleted"])
    st.markdown(
        '<p class="notice">ไฟล์ feedback ถูกเก็บที่ feedback/corrections.jsonl ใช้เป็นข้อมูลสำหรับปรับ dataset หรือ fine-tune โมเดลในรอบถัดไป</p>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is None:
    st.info("อัปโหลดภาพก่อน แล้วระบบจะให้แก้ผล AI ได้แบบรายวง")
    st.stop()

image_id = image_fingerprint(uploaded_file)
session_key = f"corrections_{image_id}_{conf_threshold}"
source_img = Image.open(uploaded_file).convert("RGB")
img_array = np.array(source_img)

with st.spinner("AI กำลังเดาเซลล์รอบแรก..."):
    # ดึงผลลัพธ์ผ่าน cached function ช่วยเพิ่มความเร็วเมื่ออินเทอร์เฟซอัปเดต
    detections = predict_blood_cells(model, img_array, conf_threshold)

if session_key not in st.session_state:
    st.session_state[session_key] = {
        str(detection["id"]): {
            "keep": detection["keep"],
            "corrected_class": detection["corrected_class"] if detection["corrected_class"] in class_options else class_options[0],
        }
        for detection in detections
    }

corrections = st.session_state[session_key]
manual_key = f"manual_detections_{image_id}_{conf_threshold}"
if manual_key not in st.session_state:
    st.session_state[manual_key] = []
manual_detections = st.session_state[manual_key]

review_source_img, review_scale = fit_review_image(source_img)
edit_col, image_col = st.columns(2, gap="large")

with edit_col:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">จัดการวงเซลล์</div>', unsafe_allow_html=True)
    st.caption("เลือกงานที่ต้องการทำ ไม่ต้องเลื่อนทั้งหน้าให้เหนื่อย")

    edit_tab, add_tab, save_tab = st.tabs(["แก้วงเดิม", "เพิ่มวงใหม่", "บันทึกบทเรียน"])

    with edit_tab:
        if not detections:
            st.warning("ยังไม่พบวงจาก AI ลองลด threshold หรือเพิ่มวงเองในแท็บถัดไป")
        else:
            tag_counts = Counter(
                corrections[str(detection["id"])]["corrected_class"]
                for detection in detections
                if str(detection["id"]) in corrections and corrections[str(detection["id"])]["keep"]
            )
            tag_options = ["ทั้งหมด"] + [
                class_name for class_name in class_options if tag_counts.get(class_name, 0) > 0
            ]
            selected_tag = st.radio(
                "กรองตามแท็ก",
                tag_options,
                horizontal=True,
                format_func=lambda tag: "ทั้งหมด" if tag == "ทั้งหมด" else f"{class_display_name(tag)} ({tag_counts[tag]})",
                key=f"tag_filter_{image_id}",
            )
            visible_detections = [
                detection
                for detection in detections
                if str(detection["id"]) in corrections and (
                    selected_tag == "ทั้งหมด"
                    or corrections[str(detection["id"])]["corrected_class"] == selected_tag
                )
            ]

            st.write(f"แสดง {len(visible_detections)} วง")
            with st.expander("แก้ทีละวง", expanded=True):
                for detection in visible_detections[:40]:
                    det_id = detection["id"]
                    key = str(det_id)
                    current = corrections[key]
                    
                    row_col1, row_col2, row_col3 = st.columns([0.85, 0.55, 1.15])
                    row_col1.write(f"#{det_id} | AI: {detection['original_class']}")
                    
                    chk_key = f"keep_{image_id}_{det_id}"
                    row_col2.checkbox(
                        "เก็บ",
                        value=current["keep"],
                        key=chk_key,
                        on_change=on_widget_change,
                        args=(session_key, det_id, "keep", chk_key)
                    )
                    
                    sel_key = f"class_{image_id}_{det_id}"
                    row_col3.selectbox(
                        "คลาส",
                        class_options,
                        index=class_options.index(current["corrected_class"]) if current["corrected_class"] in class_options else 0,
                        format_func=class_display_name,
                        key=sel_key,
                        disabled=not st.session_state.get(chk_key, current["keep"]),
                        on_change=on_widget_change,
                        args=(session_key, det_id, "corrected_class", sel_key),
                        label_visibility="collapsed",
                    )
                if len(visible_detections) > 40:
                    st.info("แสดงทีละ 40 วงเพื่อให้หน้าไม่หนัก ใช้แท็กด้านบนช่วยกรองเพิ่ม")

    with add_tab:
        st.write("เลือกคลาส แล้วลากกรอบบนภาพ")
        manual_class = st.selectbox(
            "คลาสของวงใหม่",
            class_options,
            format_func=class_display_name,
            key=f"manual_class_{image_id}",
        )
        # ปรับปรุง: เปลี่ยน realtime_update เป็น False เพื่อความเสถียร ไม่ให้โปรแกรม Crash ตอนอัปโหลดไฟล์
        crop_box = crop_box_widget(review_source_img, manual_class, image_id)

        st.caption("ลากหรือปรับกรอบบนภาพ แล้วกดเพิ่มกรอบนี้ ภาพฝั่งนี้ใช้ขนาดเดียวกับภาพหลัก")
        add_crop_box = st.button("เพิ่มกรอบนี้", type="primary")
        if add_crop_box:
            if not crop_box:
                st.warning("ยังไม่เจอกรอบบนภาพ")
            else:
                img_w, img_h = source_img.size
                x1 = max(0, float(crop_box["left"]) / review_scale)
                y1 = max(0, float(crop_box["top"]) / review_scale)
                x2 = min(img_w - 1, x1 + float(crop_box["width"]) / review_scale)
                y2 = min(img_h - 1, y1 + float(crop_box["height"]) / review_scale)

                if (x2 - x1) < 4 or (y2 - y1) < 4:
                    st.warning("กรอบเล็กเกินไป ลากให้ครอบเซลล์ชัดขึ้นอีกนิด")
                else:
                    manual_id = f"M{len(manual_detections) + 1}"
                    manual_detection = {
                        "id": manual_id,
                        "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                        "original_class": "__manual__",
                        "corrected_class": manual_class,
                        "confidence": 1.0,
                        "keep": True,
                    }
                    manual_detections.append(manual_detection)
                    corrections[str(manual_id)] = {
                        "keep": True,
                        "corrected_class": manual_class,
                    }
                    st.session_state[manual_key] = manual_detections
                    st.session_state[session_key] = corrections
                    st.rerun()

        if manual_detections:
            st.write(f"วงที่เพิ่มเอง: {len(manual_detections)} วง")
            if st.button("ล้างวงที่เพิ่มเองทั้งหมด"):
                for detection in manual_detections:
                    corrections.pop(str(detection["id"]), None)
                st.session_state[manual_key] = []
                st.session_state[session_key] = corrections
                st.warning("ล้างวงที่เพิ่มเองแล้ว")
                st.rerun()

    with save_tab:
        all_for_save = detections + manual_detections
        changed_count = 0
        for detection in all_for_save:
            key = str(detection["id"])
            if key not in corrections:
                continue
            if (
                not corrections[key]["keep"]
                or corrections[key]["corrected_class"] != detection["original_class"]
            ):
                changed_count += 1
        st.metric("บทเรียนใหม่ในภาพนี้", changed_count)
        st.caption("บันทึกแล้วจะไปอยู่ใน feedback/corrections.jsonl เพื่อเอาไปทำ dataset/train รอบต่อไป")
        save_clicked = st.button("บันทึก feedback ให้ AI เรียนต่อ", type="primary")
        if save_clicked:
            saved_count = save_feedback(
                image_id=image_id,
                file_name=uploaded_file.name,
                detections=all_for_save,
                corrections=corrections,
            )
            if saved_count:
                st.success(f"บันทึกบทเรียนใหม่ {saved_count} รายการแล้ว")
                st.rerun()
            else:
                st.info("ยังไม่มีการแก้ ลบ หรือเพิ่มวงใหม่ให้บันทึก")

    st.markdown("</div>", unsafe_allow_html=True)

all_detections = detections + manual_detections
corrected_detections = apply_corrections(all_detections, corrections)
annotated_img = draw_boxes(source_img, corrected_detections)
review_annotated_img, _ = fit_review_image(annotated_img)
rows = build_rows(corrected_detections)
counts = Counter(row["ชนิดเซลล์"] for row in rows)
risk_label, risk_class, risk_message = overall_risk(rows)
avg_confidence = round(sum(row["ความมั่นใจ"] for row in rows) / len(rows), 2) if rows else 0
high_risk_count = sum(row["ระดับความเสี่ยง"] == "สูง" for row in rows)

with image_col:
    st.markdown(f'<div class="risk-banner risk-{risk_class}">', unsafe_allow_html=True)
    st.markdown(f'<div class="risk-title">{risk_label}</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="risk-copy">{risk_message}</p>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">ภาพหลังแก้ไข</div>', unsafe_allow_html=True)
    review_b64 = image_to_base64(review_annotated_img)
    st.markdown(
        f'<img src="data:image/png;base64,{review_b64}" style="width:100%;height:auto;border-radius:6px;display:block;" />',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

metric_a, metric_b, metric_c = st.columns(3)
metric_a.metric("จำนวนที่ยืนยัน", f"{len(rows)} เซลล์")
metric_b.metric("ความมั่นใจเฉลี่ย", f"{avg_confidence:.2f}%")
metric_c.metric("รายการเสี่ยงสูง", f"{high_risk_count} จุด")

report_col, table_col = st.columns([0.72, 1.28], gap="large")

with report_col:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">สรุปหลังแก้ไข</div>', unsafe_allow_html=True)
    if not counts:
        st.warning("ทุกวงถูกลบหรือไม่มีเซลล์ที่ผ่านเกณฑ์")
    else:
        for cell_name, count in counts.most_common():
            ratio = count / len(rows)
            st.markdown(
                f"""
                <div class="cell-row">
                    <div>
                        <div class="cell-name">{cell_name}</div>
                        <div class="cell-note">{count} จาก {len(rows)} รายการ</div>
                    </div>
                    <div><span class="pill">{ratio:.0%}</span></div>
                    <div><span class="pill">พบ {count}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

with table_col:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">รายงานความเสี่ยงรายเซลล์</div>', unsafe_allow_html=True)
    if rows:
        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
            column_config={
                "ความมั่นใจ": st.column_config.NumberColumn(format="%.2f%%"),
                "คะแนนความเสี่ยง": st.column_config.ProgressColumn(
                    min_value=0,
                    max_value=100,
                    format="%d",
                ),
            },
        )
    else:
        st.write("ไม่มีข้อมูลในตาราง")
    st.markdown(
        '<p class="notice">หมายเหตุ: feedback ที่บันทึกไว้คือข้อมูลสอนโมเดลสำหรับรอบถัดไป ไม่ใช่การวินิจฉัยทางการแพทย์</p>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
