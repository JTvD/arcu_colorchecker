import cv2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, A3
from reportlab.lib.utils import ImageReader
import io
import os


def mm_to_points(mm):
    return mm * 72 / 25.4


if __name__ == "__main__":
    print("This script generates a PDF with ArUco markers.")

    # User input for page size
    page_choice = input("Choose page size (A4/A3): ").strip().upper()
    if page_choice == "A3":
        page_size = A3
        marker_size = 100
    else:
        page_size = A4
        marker_size = 50

    pdf_width, pdf_height = page_size  # points (1 point = 1/72 inch)

    # Parameters
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # Generate 4 unique markers
    markers = []
    for marker_id in range(4):
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
        is_success, buffer = cv2.imencode(".png", marker_img)
        markers.append(ImageReader(io.BytesIO(buffer)))

    # Ensure 'patterns' folder exists
    os.makedirs("patterns", exist_ok=True)

    # Create PDF
    pdf_path = os.path.join("patterns", f"arcu_markers_corners_{page_choice}.pdf")
    c = canvas.Canvas(pdf_path, pagesize=page_size)

    if page_choice == "A3":
        # Place markers around a 210x300 mm square, centered on the page
        square_width = mm_to_points(210)
        square_height = mm_to_points(300)
        x0 = (pdf_width - square_width) / 2
        y0 = (pdf_height - square_height) / 2
        positions = [
            (x0 - marker_size, y0 + square_height),  # top-left
            (x0 + square_width, y0 + square_height),  # top-right
            (x0 - marker_size, y0 - marker_size),  # bottom-left
            (x0 + square_width, y0 - marker_size),  # bottom-right
        ]
        # Draw gray rectangle of 203x292 mm centered in the same way
        rect_width = mm_to_points(203)
        rect_height = mm_to_points(292)
        rect_x = (pdf_width - rect_width) / 2
        rect_y = (pdf_height - rect_height) / 2
        c.setFillColorRGB(0.7, 0.7, 0.7)  # light gray
        c.rect(rect_x, rect_y, rect_width, rect_height, fill=1, stroke=0)
        c.setFillColorRGB(0, 0, 0)  # reset to black for any further drawing
    else:
        # Place markers around a 70x115 mm square, centered on the page
        square_width = mm_to_points(70)
        square_height = mm_to_points(115)
        x0 = (pdf_width - square_width) / 2
        y0 = (pdf_height - square_height) / 2
        positions = [
            (x0 - marker_size, y0 + square_height),  # top-left
            (x0 + square_width, y0 + square_height),  # top-right
            (x0 - marker_size, y0 - marker_size),  # bottom-left
            (x0 + square_width, y0 - marker_size),  # bottom-right
        ]
        # Draw gray rectangle of 63.5x109 mm centered in the same way
        rect_width = mm_to_points(63.5)
        rect_height = mm_to_points(109)
        rect_x = (pdf_width - rect_width) / 2
        rect_y = (pdf_height - rect_height) / 2
        c.setFillColorRGB(0.7, 0.7, 0.7)  # light gray
        c.rect(rect_x, rect_y, rect_width, rect_height, fill=1, stroke=0)
        c.setFillColorRGB(0, 0, 0)  # reset to black for any further drawing

    for img, pos in zip(markers, positions):
        c.drawImage(img, pos[0], pos[1], width=marker_size, height=marker_size, mask='auto')

    c.save()
    print(f"PDF created: {pdf_path}")
