"""
src/services/detector.py
Trạm 1: Text Detection bằng PaddleOCR 2.7.3.
Nhiệm vụ: Tìm ra vị trí (bounding boxes) của các vùng chữ trong ảnh.
"""
import cv2
import numpy as np
from paddleocr import PaddleOCR


MAX_DIMENSION = 2048  # Giới hạn max để tránh OOM trên RTX 3050


def preprocess_for_detection(image: np.ndarray) -> np.ndarray:
    """
    Tiền xử lý ảnh toàn cục trước khi đưa vào PaddleOCR:
    1. Resize ảnh lớn về max 2048px (giữ tỷ lệ)
    2. Deskew nhẹ nếu ảnh bị nghiêng (dựa trên minAreaRect)
    """
    h, w = image.shape[:2]

    # 1. Resize nếu quá lớn
    if max(h, w) > MAX_DIMENSION:
        scale = MAX_DIMENSION / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"[Detector] Resize ảnh: {w}x{h} → {new_w}x{new_h}")

    # 2. Deskew nhẹ (chỉ xoay nếu góc nghiêng nhỏ ≤ 5°)
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        coords = np.column_stack(np.where(gray < 200))  # tìm pixel text (dark)
        if len(coords) > 100:
            angle = cv2.minAreaRect(coords)[-1]
            # Chuẩn hóa góc
            if angle < -45:
                angle = 90 + angle
            elif angle > 45:
                angle = angle - 90
            # Chỉ xoay nếu góc nhỏ (tránh xoay sai ảnh dọc)
            if abs(angle) > 0.5 and abs(angle) <= 5.0:
                (ch, cw) = image.shape[:2]
                center = (cw // 2, ch // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(image, M, (cw, ch),
                                       flags=cv2.INTER_CUBIC,
                                       borderMode=cv2.BORDER_REPLICATE)
                print(f"[Detector] Deskew: xoay {angle:.1f}°")
    except Exception as e:
        print(f"[Detector] Deskew bỏ qua: {e}")

    return image


_paddle_ocr = None


def get_paddle_ocr():
    """Lazy initialization - chỉ khởi tạo lần đầu tiên được gọi."""
    global _paddle_ocr
    if _paddle_ocr is None:
        print("[Detector] Đang khởi tạo PaddleOCR 2.7.3...")
        _paddle_ocr = PaddleOCR(
            use_angle_cls=True,   # Phát hiện chữ bị nghiêng
            lang='vi',            # Tiếng Việt
            use_gpu=False,        # CPU (PaddlePaddle GPU conflict với PyTorch GPU)
            show_log=False,       # Tắt log ồn ào
        )
        print("[Detector] PaddleOCR đã sẵn sàng!")
    return _paddle_ocr


def detect_text_regions(image: np.ndarray) -> list:
    """
    Phát hiện các vùng chứa chữ trong ảnh.

    Args:
        image: Ảnh dạng numpy array (BGR - đọc từ OpenCV)

    Returns:
        List các bounding boxes, mỗi phần tử là list 4 điểm tọa độ:
        [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], ...]
    """
    # ★ Tiền xử lý trước detection
    image = preprocess_for_detection(image)

    ocr = get_paddle_ocr()
    result = ocr.ocr(image, cls=True)

    if not result or result[0] is None:
        return []

    boxes = []
    for line in result[0]:
        # line = [box_coordinates, (text, confidence)]
        box = line[0]
        confidence = line[1][1]
        if confidence > 0.5:
            boxes.append(box)

    return boxes


def detect_with_text(image: np.ndarray) -> list:
    """
    Phát hiện vùng chữ VÀ đọc text bằng PaddleOCR một lượt.
    Trả về List các tuple (box, text, confidence).
    """
    ocr = get_paddle_ocr()
    result = ocr.ocr(image, cls=True)

    if not result or result[0] is None:
        return []

    items = []
    for line in result[0]:
        box = line[0]
        text = line[1][0]
        confidence = line[1][1]
        if confidence > 0.5:
            items.append((box, text, confidence))

    return items
