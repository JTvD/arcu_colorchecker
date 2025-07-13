import cv2
import os
import numpy as np
from scipy.spatial import ConvexHull
import pandas as pd

from webcam import take_image
import helpers as helpers


if __name__ == "__main__":
    # Try to load test image, otherwise use webcam to capture one
    # img_path = 'example_imgs/color_checker (7).jpg'
    # settings = helpers.SquareConfig(75, [175, 325], 90)
    img_path = 'example_imgs/color_checker_mini (1).jpg'
    settings = helpers.SquareConfig(40, [195, 335], 170)

    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w * 0.5), int(h * 0.5)), interpolation=cv2.INTER_AREA)
        print(f"Loaded image: {img_path}")
    else:
        img = take_image()
    # Detect ArUco markers in the image
    img_markers, corners, ids, rejected = helpers.find_arcu_markers(img)
    id_tuples = helpers.get_max_marker_distance(corners, ids)

    if corners is None or len(corners) < 4 or len(id_tuples) < 1:
        exit("Not enough markers detected or not enough marker pairs found.")

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
        dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]],
                       dtype=np.float32)
        H = cv2.getPerspectiveTransform(ordered, dst)

    # Store the color values of the squares with their centers & distances
    square_values = []
    # Between the furtherth away markers
    row_id = 0
    for id1, id2 in id_tuples:
        # Get marker centers in the rectified plane
        # Map image marker centers to rectified plane using inverse perspective
        # (ordered: image, dst: rectified)

        marker_centers_img = [np.mean(c[0], axis=0) for c in corners]
        marker_centers_rect = cv2.perspectiveTransform(np.array(marker_centers_img,
                                                                dtype=np.float32)
                                                        [None, :, :], H)[0]
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
        region_start = start + direction_norm * settings.region_offset
        region_end = end - direction_norm * settings.region_offset
        # Compute region center in rectified plane
        region_center = (region_start + region_end) / 2
        # helpers.draw_perpendicular_line(img_markers, ordered, dst,
        #                                 region_start, region_end, perp_direction)

        for shift_amount in settings.shift_amount:  # Shift toward the center of the image
            center_plus = region_center + perp_direction * shift_amount
            center_minus = region_center - perp_direction * shift_amount
            center_plus_img = helpers.perspective_transform_points(ordered, dst,
                                                                    center_plus[None, :])[0]
            center_minus_img = helpers.perspective_transform_points(ordered, dst,
                                                                    center_minus[None, :])[0]
            h_img, w_img = img_markers.shape[:2]
            # Determine the shift sign based on distance to edges
            if helpers.min_dist_to_edge(center_plus_img, h_img, w_img) >\
               helpers.min_dist_to_edge(center_minus_img, h_img, w_img):
                shift_sign = 1
            else:
                shift_sign = -1
            for i in range(6):
                frac = (i + 0.5) / 6
                center = region_start + (region_end - region_start) * frac
                center_shifted = center + perp_direction * shift_sign * shift_amount
                square_rect = np.array([
                    [center_shifted[0] - settings.square_size//2,
                        center_shifted[1] - settings.square_size//2],
                    [center_shifted[0] + settings.square_size//2,
                        center_shifted[1] - settings.square_size//2],
                    [center_shifted[0] + settings.square_size//2,
                        center_shifted[1] + settings.square_size//2],
                    [center_shifted[0] - settings.square_size//2,
                        center_shifted[1] + settings.square_size//2]
                ], dtype=np.float32)
                # Map these points back to the image perspective
                square_img = helpers.perspective_transform_points(ordered, dst, square_rect)
                pts_int = square_img.astype(int).reshape((-1, 1, 2))
                cv2.polylines(img_markers, [pts_int], isClosed=True, color=(0, 255, 0),
                              thickness=2)
                # Map label position as well
                label_img = helpers.perspective_transform_points(ordered, dst,
                                                                 center_shifted[None, :])[0]
                # cv2.putText(img_markers, f"S{i+1}_{shift_amount}", (int(label_img[0]),
                #             int(label_img[1]) - 10),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                # Compute average color inside the square
                mask = np.zeros(img_markers.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [pts_int], 255)
                mean_val = cv2.mean(img, mask=mask)[:3]  # BGR
                square_values.append({
                    'center': center_shifted,
                    'color': mean_val,
                    'row_id': row_id,
                    'col_id': i
                })
            row_id += 1
    # After collecting all square_values
    df = pd.DataFrame(square_values)
    # Flip row_id 2 & 3 before sorting: shift_amount first does the outer column
    df.loc[df['row_id'] == 2, 'row_id'] = -1
    df.loc[df['row_id'] == 3, 'row_id'] = 2
    df.loc[df['row_id'] == -1, 'row_id'] = 3
    df_sorted = df.sort_values(by=["row_id", "col_id"])

    # Draw numbers at the center of each sorted square
    i = 1
    for idx, row in df_sorted.iterrows():
        center = row['center']
        # Map the rectified center back to the image perspective
        center_img = helpers.perspective_transform_points(ordered, dst, np.array([center]))[0]
        cv2.putText(
            img_markers,
            str(i),
            (int(center_img[0]), int(center_img[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )
        i += 1

    # TODO, could still be 180 degrees rotated > easy check if looking for black/white
    # TODO: Write conversion function

    cv2.imshow("Warped Image", img_markers)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
