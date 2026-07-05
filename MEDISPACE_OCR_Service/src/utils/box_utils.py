"""
src/utils/box_utils.py
Các hàm tiện ích để xử lý và sắp xếp Bounding Boxes từ PaddleOCR.
"""
import numpy as np
import cv2


def get_center_y(box):
    """Tính tọa độ Y phía trên của một bounding box (top edge) để sort chuẩn hơn."""
    ys = [point[1] for point in box]
    return min(ys)  # Dùng cạnh trên (top edge) thay vì center để tránh out-of-order



def get_center_x(box):
    """Tính tọa độ X trung bình của một bounding box."""
    xs = [point[0] for point in box]
    return sum(xs) / len(xs)


def sort_boxes_by_reading_order(boxes_with_text: list, line_threshold: int = 15) -> list:
    """
    Sắp xếp danh sách (box, text) theo thứ tự đọc tự nhiên:
    - Từ trên xuống dưới (theo trục Y)
    - Trong cùng một hàng: từ trái qua phải (theo trục X)

    Args:
        boxes_with_text: List các tuple (box, text)
        line_threshold: Ngưỡng pixel để xác định 2 box có cùng hàng

    Returns:
        List các tuple (box, text) đã được sắp xếp
    """
    if not boxes_with_text:
        return []

    sorted_items = sorted(boxes_with_text, key=lambda item: get_center_y(item[0]))

    lines = []
    current_line = [sorted_items[0]]

    for item in sorted_items[1:]:
        current_y = get_center_y(item[0])
        first_y = get_center_y(current_line[0][0])  # ★ Sửa lỗi trôi Y: neo chặt vào item ĐẦU TIÊN của dòng

        if abs(current_y - first_y) <= line_threshold:
            current_line.append(item)
        else:
            lines.append(current_line)
            current_line = [item]

    lines.append(current_line)

    result = []
    for line in lines:
        line_sorted = sorted(line, key=lambda item: get_center_x(item[0]))
        result.extend(line_sorted)

    return result


def _order_box_points(points: np.ndarray) -> np.ndarray:
    """Order 4 box points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect

def _crop_rectangle_with_padding(image: np.ndarray, box: np.ndarray, padding: int = 4) -> np.ndarray:
    x_min = max(0, int(np.min(box[:, 0])) - padding)
    x_max = min(image.shape[1], int(np.max(box[:, 0])) + padding)
    y_min = max(0, int(np.min(box[:, 1])) - padding)
    y_max = min(image.shape[0], int(np.max(box[:, 1])) + padding)
    return image[y_min:y_max, x_min:x_max]

def crop_image_by_box(image: np.ndarray, box) -> np.ndarray:
    """
    Cắt ảnh theo bounding box từ PaddleOCR.

    Args:
        image: Ảnh numpy array (BGR từ OpenCV)
        box: Danh sách 4 điểm tọa độ [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]

    Returns:
        Ảnh đã được cắt dạng numpy array
    """
    box = np.array(box, dtype=np.float32)
    padding = 4

    try:
        rect = _order_box_points(box)
        (tl, tr, br, bl) = rect
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_width = int(max(width_a, width_b))
        max_height = int(max(height_a, height_b))

        if max_width >= 2 and max_height >= 2:
            dst = np.array(
                [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
                dtype="float32",
            )
            matrix = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image, matrix, (max_width, max_height), flags=cv2.INTER_CUBIC)
            return cv2.copyMakeBorder(
                warped,
                padding,
                padding,
                padding,
                padding,
                borderType=cv2.BORDER_REPLICATE,
            )
    except Exception:
        pass

    box = box.astype(np.int32)
    x_min = max(0, int(np.min(box[:, 0])))
    x_max = min(image.shape[1], int(np.max(box[:, 0])))
    y_min = max(0, int(np.min(box[:, 1])))
    y_max = min(image.shape[0], int(np.max(box[:, 1])))

    # Thêm padding nhỏ để VietOCR đọc chuẩn hơn
    padding = 4
    y_min = max(0, y_min - padding)
    y_max = min(image.shape[0], y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(image.shape[1], x_max + padding)

    return image[y_min:y_max, x_min:x_max]
