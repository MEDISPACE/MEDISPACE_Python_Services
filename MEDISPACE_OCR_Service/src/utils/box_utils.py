"""
src/utils/box_utils.py
Các hàm tiện ích để xử lý và sắp xếp Bounding Boxes từ PaddleOCR.
"""
import numpy as np


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


def crop_image_by_box(image: np.ndarray, box) -> np.ndarray:
    """
    Cắt ảnh theo bounding box từ PaddleOCR.

    Args:
        image: Ảnh numpy array (BGR từ OpenCV)
        box: Danh sách 4 điểm tọa độ [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]

    Returns:
        Ảnh đã được cắt dạng numpy array
    """
    box = np.array(box, dtype=np.int32)
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
