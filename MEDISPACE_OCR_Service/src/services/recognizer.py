"""
src/services/recognizer.py
Trạm 2: Text Recognition bằng VietOCR.
Nhiệm vụ: Đọc chính xác chữ Tiếng Việt từ các vùng ảnh đã được cắt.
"""
import cv2
import os
import torch
import numpy as np
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

from src.utils.box_utils import sort_boxes_by_reading_order, crop_image_by_box


# ─── Image Preprocessing ─────────────────────────────────────────────────────

MIN_HEIGHT = 32  # VietOCR chính xác hơn khi ảnh ≥ 32px cao

def preprocess_crop(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Tiền xử lý vùng crop trước khi đưa vào VietOCR.
    Pipeline:
      1. Grayscale
      2. CLAHE (tăng contrast cục bộ — hiệu quả cho ảnh chụp đèn vàng / mờ)
      3. Bilateral filter (giảm noise nhưng giữ cạnh chữ)
      4. Resize nếu quá nhỏ (VietOCR cần ít nhất 32px height)
      5. Convert lại RGB 3-channel cho VietOCR
    """
    h, w = crop_bgr.shape[:2]
    if h == 0 or w == 0:
        return crop_bgr

    # 1. Grayscale
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    # 2. CLAHE — Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    # 3. Bilateral filter — giảm noise nhưng giữ nguyên cạnh chữ
    filtered = cv2.bilateralFilter(enhanced, d=5, sigmaColor=50, sigmaSpace=50)

    # 4. Resize nếu ảnh quá nhỏ
    if h < MIN_HEIGHT:
        scale = MIN_HEIGHT / h
        new_w = max(int(w * scale), 1)
        filtered = cv2.resize(filtered, (new_w, MIN_HEIGHT), interpolation=cv2.INTER_CUBIC)

    # 5. Convert grayscale → RGB 3-channel (VietOCR expects RGB)
    rgb = cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)
    return rgb


# Lazy initialization
_vietocr_detector = None


def _get_device() -> str:
    """Tự động chọn device phù hợp: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        return 'cuda:0'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def get_vietocr():
    """Lazy initialization - chỉ khởi tạo lần đầu tiên được gọi."""
    global _vietocr_detector
    if _vietocr_detector is None:
        device = _get_device()
        print(f"[Recognizer] Dang khoi tao VietOCR model tren {device.upper()}...")
        config = Cfg.load_config_from_name('vgg_transformer')
        config['device'] = device
        config['predictor']['beamsearch'] = os.getenv("VIETOCR_BEAMSEARCH", "true").lower() != "false"
        _vietocr_detector = Predictor(config)
        print(f"[Recognizer] VietOCR đã sẵn sàng! (device={device}, beamsearch={config['predictor']['beamsearch']})")
    return _vietocr_detector


def move_vietocr_to_cpu():
    """Đẩy VietOCR model xuống CPU để giải phóng VRAM cho Ollama."""
    global _vietocr_detector
    if _vietocr_detector is not None and torch.cuda.is_available():
        _vietocr_detector.model.to('cpu')
        _vietocr_detector.config['device'] = 'cpu'
        torch.cuda.empty_cache()
        print("[Recognizer] ↓ VietOCR đã chuyển xuống CPU (giải phóng VRAM)")


def move_vietocr_to_gpu():
    """Kéo VietOCR model lên GPU để sẵn sàng cho request tiếp theo."""
    global _vietocr_detector
    if _vietocr_detector is not None and torch.cuda.is_available():
        _vietocr_detector.model.to('cuda:0')
        _vietocr_detector.config['device'] = 'cuda:0'
        print("[Recognizer] ↑ VietOCR đã chuyển lên GPU")


def recognize_text_from_boxes(image: np.ndarray, boxes: list) -> list:
    """
    Nhận diện text Tiếng Việt từ danh sách bounding boxes.
    Đây là điểm kết nối giữa Trạm 1 (PaddleOCR) và Trạm 2 (VietOCR).

    Args:
        image: Ảnh gốc dạng numpy array (BGR từ OpenCV)
        boxes: List các bounding boxes từ detector.py

    Returns:
        List các tuple (box, text) đã được sắp xếp theo thứ tự đọc
    """
    if not boxes:
        return []

    vietocr = get_vietocr()
    boxes_with_text = []

    for box in boxes:
        # Cắt vùng ảnh theo bounding box
        cropped_bgr = crop_image_by_box(image, box)

        if cropped_bgr.size == 0:
            continue

        try:
            # ★ Tiền xử lý: CLAHE + bilateral filter + resize
            processed_rgb = preprocess_crop(cropped_bgr)

            # VietOCR cần ảnh PIL định dạng RGB
            cropped_pil = Image.fromarray(processed_rgb)  # đã là RGB từ preprocess_crop

            # Nhận diện text
            text = vietocr.predict(cropped_pil)
            text = text.strip()

            if text:  # Chỉ giữ lại kết quả có nội dung
                boxes_with_text.append((box, text))
        except Exception as e:
            print(f"[Recognizer] Lỗi khi nhận diện 1 vùng: {e}")
            continue

    # Sắp xếp theo thứ tự đọc tự nhiên (trên→dưới, trái→phải)
    sorted_items = sort_boxes_by_reading_order(boxes_with_text)

    return sorted_items


def extract_full_text(image: np.ndarray, boxes: list) -> str:
    """
    Trích xuất toàn bộ text từ ảnh thành một chuỗi string liên tục.
    Dùng để truyền vào Trạm 3 (Extractor).

    Args:
        image: Ảnh gốc dạng numpy array
        boxes: Danh sách bounding boxes từ detector

    Returns:
        Chuỗi text đầy đủ của đơn thuốc, các dòng ngăn cách bởi newline
    """
    items = recognize_text_from_boxes(image, boxes)
    lines = [text for _, text in items]
    return "\n".join(lines)

