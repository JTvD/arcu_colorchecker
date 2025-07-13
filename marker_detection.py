import cv2
import os
import numpy as np
from scipy.spatial import ConvexHull

from webcam import take_image


def find_arcu_markers(img):
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
    # Show the detected markers image for debugging
    # cv2.imshow('Detected Markers', img_markers)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img_markers, corners, ids, rejected


def get_max_marker_distance(corners, ids):
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


def get_marker_size(corners, ids, marker_id):
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


def perspective_transform_points(src_pts, dst_pts, points):
    """
    Transforms a set of points from the destination (ideal/rectified) plane to the source (image) plane using a perspective transform.
    Args:
        src_pts (np.ndarray): 4x2 array of source (image) points (corners in the image).
        dst_pts (np.ndarray): 4x2 array of destination (rectified) points (corners in the ideal plane).
        points (np.ndarray): Nx2 array of points in the destination plane to transform to the image plane.
    Returns:
        np.ndarray: Nx2 array of transformed points in the image plane.
    """
    H = cv2.getPerspectiveTransform(dst_pts, src_pts)
    points_h = np.hstack([points, np.ones((points.shape[0], 1))])
    points_img_h = (H @ points_h.T).T
    points_img = points_img_h[:, :2] / points_img_h[:, 2:]
    return points_img


def draw_perpendicular_line(img, ordered, dst, region_start, region_end, perp_direction, color=(255, 0, 0), line_len=200, thickness=2):
    """
    Draw a line in the direction of perp_direction at the center between region_start and region_end.
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


def min_dist_to_edge(pt):
    """ Calculates the minimum distance from a point to the edges of the image.
    Args:
        pt (tuple): A point (x, y) in the image.
    Returns:
        int: Minimum distance to the image edges.
    """
    x, y = pt
    return min(x, w_img - x, y, h_img - y)


if __name__ == "__main__":
    # Try to load test image, otherwise use webcam to capture one
    img_path = 'example_imgs/color_checker (7).jpg'
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w * 0.5), int(h * 0.5)), interpolation=cv2.INTER_AREA)
        print(f"Loaded image: {img_path}")
    else:
        img = take_image()
    # Detect ArUco markers in the image
    img_markers, corners, ids, rejected = find_arcu_markers(img)
    id_tuples = get_max_marker_distance(corners, ids)

    if corners is not None and len(corners) >= 4 and len(id_tuples) > 0:
        # Get all marker centers
        marker_centers = [np.mean(c[0], axis=0) for c in corners]
        hull = ConvexHull(marker_centers)
        hull_indices = hull.vertices
        if len(hull_indices) >= 4:
            pts = np.array([marker_centers[i] for i in hull_indices], dtype=np.float32)
            s = pts.sum(axis=1)
            diff = np.diff(pts, axis=1)
            ordered = np.zeros((4, 2), dtype=np.float32)
            ordered[0] = pts[np.argmin(s)]      # top-left
            ordered[2] = pts[np.argmax(s)]      # bottom-right
            ordered[1] = pts[np.argmin(diff)]   # top-right
            ordered[3] = pts[np.argmax(diff)]   # bottom-left
            width = 800
            height = int(width * 297 / 210)  # A4 aspect
            dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype=np.float32)
            # size of the square in the rectified plane
            square_size = 75

            # Between the furtherth away markers
            for id1, id2 in id_tuples:
                # Get marker centers in the rectified plane
                # Map image marker centers to rectified plane using inverse perspective
                # (ordered: image, dst: rectified)
                H = cv2.getPerspectiveTransform(ordered, dst)
                marker_centers_img = [np.mean(c[0], axis=0) for c in corners]
                marker_centers_rect = cv2.perspectiveTransform(np.array(marker_centers_img, dtype=np.float32)[None, :, :], H)[0]
                # Find the rectified positions for id1 and id2
                idx1 = np.where(ids.flatten() == id1)[0][0]
                idx2 = np.where(ids.flatten() == id2)[0][0]
                start = marker_centers_rect[idx1]
                end = marker_centers_rect[idx2]
                # Compute direction vector for offset (along the line from start to end)
                direction = end - start
                direction_norm = direction / np.linalg.norm(direction)
                perp_direction = np.array([-direction_norm[1], direction_norm[0]])

                # Define the valid region for square centers (50px from each marker)
                region_start = start + direction_norm * 90  # Shift from marker 1
                region_end = end - direction_norm * 90  # Shift from marker 2
                # Compute region center in rectified plane
                region_center = (region_start + region_end) / 2
                # draw_perpendicular_line(img_markers, ordered, dst, region_start, region_end, perp_direction)

                for shift_amount in [175, 325]:  # Shift toward the center of the image
                    center_plus = region_center + perp_direction * shift_amount
                    center_minus = region_center - perp_direction * shift_amount
                    center_plus_img = perspective_transform_points(ordered, dst, center_plus[None, :])[0]
                    center_minus_img = perspective_transform_points(ordered, dst, center_minus[None, :])[0]
                    h_img, w_img = img_markers.shape[:2]
                    # Determine the shift sign based on distance to edges
                    if min_dist_to_edge(center_plus_img) > min_dist_to_edge(center_minus_img):
                        shift_sign = 1
                    else:
                        shift_sign = -1
                    for i in range(6):
                        frac = (i + 0.5) / 6
                        center = region_start + (region_end - region_start) * frac
                        center_shifted = center + perp_direction * shift_sign * shift_amount
                        square_rect = np.array([
                            [center_shifted[0] - square_size//2, center_shifted[1] - square_size//2],
                            [center_shifted[0] + square_size//2, center_shifted[1] - square_size//2],
                            [center_shifted[0] + square_size//2, center_shifted[1] + square_size//2],
                            [center_shifted[0] - square_size//2, center_shifted[1] + square_size//2]
                        ], dtype=np.float32)
                        # Map these points back to the image perspective
                        square_img = perspective_transform_points(ordered, dst, square_rect)
                        pts_int = square_img.astype(int).reshape((-1, 1, 2))
                        cv2.polylines(img_markers, [pts_int], isClosed=True, color=(0, 255, 0), thickness=2)
                        # Map label position as well
                        label_img = perspective_transform_points(ordered, dst, center_shifted[None, :])[0]
                        #cv2.putText(img_markers, f"S{i+1}_{shift_amount}", (int(label_img[0]), int(label_img[1]) - 10),
                        #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Warped Image", img_markers)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
