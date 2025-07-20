from dataclasses import dataclass
import cv2
import numpy as np
import pandas as pd


@dataclass
class SquareConfig:
    square_size: int = 75
    # List of shift amounts for the squares in the rectified plane
    shift_amount: list = None
    # Offset from Marker 1/2
    region_offset: int = 90

    def __init__(self, square_size: int = 75, shift_amount=None, region_offset: int = 90):
        """Initializes the SquareConfig with the given parameters.
        """
        self.square_size = square_size
        if shift_amount is None:
            self.shift_amount = [175, 325]
        else:
            self.shift_amount = shift_amount
        self.region_offset = region_offset


def find_arcu_markers(img: np.ndarray) -> tuple:
    """
    Detects ArUco markers in the given image.
    Args:
        img (numpy.ndarray): The input image in which to detect markers.
    Returns:
        tuple: Detected corners, ids, and rejected candidates.
    """
    # Define the ArUco dictionary and parameters
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()

    # Create an ArUco detector
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # Detect markers
    corners, ids, rejected = detector.detectMarkers(img)
    if ids is not None and len(ids) > 0:
        print(f"Detected marker IDs: {ids.flatten()}")
        img_markers = cv2.aruco.drawDetectedMarkers(img.copy(), corners, ids)
    else:
        print("No ArUco markers detected.")
        img_markers = img.copy()  # fallback to original image if no markers
    return img_markers, corners, ids, rejected


def get_max_marker_distance(corners: list, ids: np.ndarray) -> list:
    """
    Finds the two marker pairs with the 3rd and 4th largest distances between their centers.
    Args:
        corners (list): List of marker corners.
        ids (numpy.ndarray): Marker IDs.
    Returns:
        list: [(id1, id2), ...] for 3rd and 4th largest distances.
    """
    if ids is None or len(ids) < 2:
        print("Not enough markers detected for distance calculation.")
        return []
    centers = [np.mean(c[0], axis=0) for c in corners]
    dists = {}
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dist = np.linalg.norm(centers[i] - centers[j])
            dists[dist] = (i, j)
    # Sort distances descending
    sorted_dists = sorted(dists.keys(), reverse=True)
    results = []
    for idx in [2, 3]:  # 3rd and 4th highest (0-based)
        i, j = dists[sorted_dists[idx]]
        results.append((ids[i][0], ids[j][0]))
    return results


def get_marker_size(corners: list, ids: np.ndarray, marker_id: int) -> tuple:
    """
    Returns the width and height of the marker (distance between corners) for the given ID.
    Args:
        corners (list): List of marker corners.
        ids (numpy.ndarray): Marker IDs.
        marker_id (int): The ID of the marker to find.
    Returns:
        tuple: (width, height) in pixels, or None if not found.
    """
    if ids is None:
        return None
    for i, id_val in enumerate(ids.flatten()):
        if id_val == marker_id:
            pts = corners[i][0]
            width = np.linalg.norm(pts[0] - pts[1])
            height = np.linalg.norm(pts[1] - pts[2])
            return width, height
    return None


def perspective_transform_points(src_pts: np.ndarray, dst_pts: np.ndarray,
                                 points: np.ndarray) -> np.ndarray:
    """
    Transforms a set of points from the destination (ideal/rectified) plane to
    the source (image) plane using a perspective transform.
    Args:
        src_pts (np.ndarray): 4x2 array of source (image) points
                              (corners in the image).
        dst_pts (np.ndarray): 4x2 array of destination (rectified) points
                              (corners in the ideal plane).
        points (np.ndarray): Nx2 array of points in the destination plane to
                             transform to the image plane.
    Returns:
        np.ndarray: Nx2 array of transformed points in the image plane.
    """
    H = cv2.getPerspectiveTransform(dst_pts, src_pts)
    points_h = np.hstack([points, np.ones((points.shape[0], 1))])
    points_img_h = (H @ points_h.T).T
    points_img = points_img_h[:, :2] / points_img_h[:, 2:]
    return points_img


def draw_perpendicular_line(img: np.ndarray, ordered: np.ndarray, dst: np.ndarray,
                            region_start: np.ndarray, region_end: np.ndarray,
                            perp_direction: np.ndarray, color=(255, 0, 0),
                            line_len=200, thickness=2):
    """
    Draw a line in the direction of perp_direction at the center between
    region_start and region_end.
    The line is drawn in the rectified plane and mapped to the image.
    Args:
        img (numpy.ndarray): The image on which to draw the line.
        ordered (np.ndarray): 4x2 array of ordered points in the image plane.
        dst (np.ndarray): 4x2 array of destination points in the rectified plane.
        region_start (np.ndarray): Start point of the region in the rectified plane.
        region_end (np.ndarray): End point of the region in the rectified plane.
        perp_direction (np.ndarray): Direction vector perpendicular to the line.
        color (tuple): Color of the line in BGR format.
        line_len (int): Length of the line to draw.
        thickness (int): Thickness of the line.
    """
    region_center = (region_start + region_end) / 2
    line_start_rect = region_center - perp_direction * line_len // 2
    line_end_rect = region_center + perp_direction * line_len // 2
    line_start_img = perspective_transform_points(ordered, dst, line_start_rect[None, :])[0]
    line_end_img = perspective_transform_points(ordered, dst, line_end_rect[None, :])[0]
    cv2.line(img, (int(line_start_img[0]), int(line_start_img[1])),
                  (int(line_end_img[0]), int(line_end_img[1])), color, thickness)


def min_dist_to_edge(pt: tuple, w_img: int, h_img: int) -> int:
    """ Calculates the maximum distance from a point to the edges of the image.
    Args:
        pt (tuple): A point (x, y) in the image.
        w_img (int): Width of the image.
        h_img (int): Height of the image.
    Returns:
        int: Maximum distance to the image edges.
    """
    x, y = pt
    return max(x, w_img - x, y, h_img - y)


def draw_square_numbers(img_markers: np.ndarray, df_sorted: pd.DataFrame,
                        ordered: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """ Draw numbers at the center of each sorted square on the image.
    Args:
        img_markers (np.ndarray): The image to draw on.
        df_sorted (pd.DataFrame): DataFrame with 'center' column for square centers.
        ordered (np.ndarray): 4x2 array of ordered points in the image plane.
        dst (np.ndarray): 4x2 array of destination points in the rectified plane.
    """
    for idx, row in df_sorted.iterrows():
        center = row['center']
        center_img = perspective_transform_points(ordered, dst, np.array([center]))[0]
        cv2.putText(
            img_markers, str(idx), (int(center_img[0]), int(center_img[1])),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return img_markers


def orient_color_checker(df_sorted: pd.DataFrame) -> pd.DataFrame:
    """ Detects the black square in the color checker and orients the DataFrame
    so that the black square is in position 0.
    Adds 'rgb_diff' and 'rgb_sum' columns if not present.
    Args:
        df_sorted (pd.DataFrame): DataFrame with 'color' column.
    Returns:
        pd.DataFrame: Oriented DataFrame.
    """
    # Compute rgb_diff and rgb_sum if not already present
    if 'rgb_diff' not in df_sorted.columns:
        df_sorted['rgb_diff'] = df_sorted['color'].apply(lambda rgb: np.max(rgb) - np.min(rgb))
    if 'rgb_sum' not in df_sorted.columns:
        df_sorted['rgb_sum'] = df_sorted['color'].apply(lambda rgb: np.sum(rgb))

    idx_lowest_diff = df_sorted['rgb_diff'].idxmin()
    idx_lowest_sum = df_sorted['rgb_sum'].idxmin()
    if idx_lowest_diff != idx_lowest_sum or idx_lowest_diff not in [0, 23]:
        print(f"something went wrong, the black square should be in position 1 or 24, "
              f"but found at {idx_lowest_diff} (diff) and {idx_lowest_sum} (sum).")
    else:
        print(f"black square found at position {idx_lowest_diff + 1}")

    # Flip if needed so black is in position 23
    if idx_lowest_diff == 0:
        df_sorted = df_sorted.iloc[::-1].reset_index(drop=True)
        print("Flipping the color checker horizontally to match the expected orientation.")
    return df_sorted
