import cv2
import numpy as np
import pandas as pd


def bgr_to_lab(bgr_tuple: tuple | np.ndarray) -> tuple | np.ndarray:
    """ Converts a BGR tuple to CIE LAB color space.
    LAB (L: 0-100, a/b: -128 to 127)
    Args:
        bgr_tuple (tuple|np.ndarray): BGR color tuple or ndarray.
    Returns:
        lab (tuple|np.ndarray): LAB color tuple or ndarray (L, a, b).
    Note, the references values are floats therefore the image should be converted first
    """
    bgr = np.asarray(bgr_tuple, dtype=np.float32)
    if bgr.ndim == 1:
        bgr = bgr[None, None, :]
    elif bgr.ndim == 2:
        bgr = bgr[None, :, :]
    bgr = np.clip(bgr, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    # Convert to CIE LAB: L [0-100], a/b [-128,127]
    lab_cie = np.empty_like(lab, dtype=np.float32)
    lab_cie[..., 0] = lab[..., 0] * 100.0 / 255.0
    lab_cie[..., 1] = lab[..., 1] - 128.0
    lab_cie[..., 2] = lab[..., 2] - 128.0
    if lab_cie.shape[0:2] == (1, 1):
        return tuple(lab_cie[0, 0])
    return lab_cie


def lab_to_bgr(lab_tuple: tuple | np.ndarray) -> tuple | np.ndarray:
    """Converts a LAB color tuple to an RGB color tuple using OpenCV.
    Args:
        lab_tuple (tuple|np.ndarray): LAB color tuple or ndarray (L, a, b).
    Returns:
        tuple|np.ndarray: BGR color tuple or ndarray.
    """
    # OpenCV expects LAB in uint8, so clip and convert
    lab = np.asarray(lab_tuple, dtype=np.float32)
    if lab.ndim == 1:
        lab = lab[None, None, :]
    elif lab.ndim == 2:
        lab = lab[None, :, :]
    # Scale to OpenCV LAB range: L [0,100] -> [0,255], a/b [-128,127] -> [0,255]
    lab_cv = np.empty_like(lab)
    lab_cv[..., 0] = np.clip(lab[..., 0], 0, 100) * 255.0 / 100.0
    lab_cv[..., 1] = np.clip(lab[..., 1], -128, 127) + 128.0
    lab_cv[..., 2] = np.clip(lab[..., 2], -128, 127) + 128.0
    lab_cv = np.clip(lab_cv, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(lab_cv, cv2.COLOR_LAB2BGR)
    if bgr.shape[0:2] == (1, 1):
        return tuple(bgr[0, 0])
    return bgr


def fit_color_correction_matrix(measured_lab: np.ndarray, reference_lab: np.ndarray):
    """
    Fit a 3x4 color correction matrix for CIE LAB values (L: 0-100, a/b: -128 to 127).
    Supports both linear and (optionally) polynomial correction.
    Args:
        measured_lab (np.ndarray): Measured LAB values from the camera.
        reference_lab (np.ndarray): Reference LAB values (ground truth).
    Returns:
        M (3, 4) float64 (linear) or (3, n) for polynomial
    """
    measured_lab = np.asarray(measured_lab, dtype=np.float64)
    reference_lab = np.asarray(reference_lab, dtype=np.float64)
    # Linear correction: [L, a, b, 1]
    L = measured_lab[:, 0]
    a = measured_lab[:, 1]
    b = measured_lab[:, 2]
    X = np.column_stack([
        L, a, b, np.ones_like(L)
    ])  # shape (N, 4)
    M = np.zeros((3, 4), dtype=np.float64)
    for i in range(3):
        M[i], _, _, _ = np.linalg.lstsq(X, reference_lab[:, i], rcond=None)
    return M


def apply_color_correction(lab: tuple | np.ndarray, M: np.ndarray):
    """    Apply the color correction matrix M to a single LAB color.
    Args:
        lab (tuple | np.ndarray): LAB color (L, a, b).
        M (np.ndarray): 3x4 color correction matrix.
    Returns:
        tuple: Corrected LAB color (L', a', b')
    """
    # Linear feature expansion to match fit_color_correction_matrix
    L, a, b = lab
    features = np.array([L, a, b, 1.0], dtype=np.float64)
    corrected = M @ features
    return tuple(corrected)


def apply_color_correction_to_image(img_bgr: np.ndarray, M: np.ndarray):
    """
    Apply the color correction matrix M to an entire BGR image in LAB space.
    Args:
        img_bgr (np.ndarray): Input image in BGR format, dtype uint8 or float32 [0,255].
        M (np.ndarray): 3x4 color correction matrix.
    Returns:
        np.ndarray: Color-corrected image in BGR format, dtype uint8.
    """
    # Convert BGR image to CIE LAB using universal function
    img_lab_cie = bgr_to_lab(img_bgr)
    h, w, _ = img_lab_cie.shape
    img_lab_flat = img_lab_cie.reshape(-1, 3)
    # Linear feature expansion for each pixel (L, a, b, 1)
    L = img_lab_flat[:, 0]
    a = img_lab_flat[:, 1]
    b = img_lab_flat[:, 2]
    X = np.column_stack([
        L, a, b, np.ones_like(L)
    ])  # shape (N, 4)
    img_lab_corr = (M @ X.T).T
    img_lab_corr = np.clip(img_lab_corr, [0, -128, -128], [100, 127, 127])
    img_lab_corr_img = img_lab_corr.reshape(h, w, 3)
    # Convert corrected LAB back to BGR using universal function
    img_bgr_corr = lab_to_bgr(img_lab_corr_img)
    return img_bgr_corr


def delta_e_cie76(lab1, lab2):
    """Compute the CIE76 delta E color difference between two LAB colors.
    This is a simple Euclidean distance in LAB color space, used as error metric.
    Args:
        lab1 (tuple | np.ndarray): First LAB color (L, a, b).
        lab2 (tuple | np.ndarray): Second LAB color (L, a, b).
    Returns:
        float: Delta E color difference.
    """
    lab1 = np.array(lab1)
    lab2 = np.array(lab2)
    return np.linalg.norm(lab1 - lab2)


def create_color_correction(df_colorchecker: pd.DataFrame):
    """ Creates a color correction DataFrame from the color checker DataFrame.
    Args:
        df_colorchecker (pd.DataFrame): DataFrame containing color checker data.
    Returns:
        color correction values.
    """

    reference_value_df = pd.read_csv('reference_values.csv')
    # Sort by 'id' column to ensure correct order
    reference_value_df = reference_value_df.sort_values(by='id').reset_index(drop=True)
    # Combine Lab_l, Lab_a, Lab_b columns into a single 'lab' column as a tuple
    reference_value_df['lab'] = reference_value_df.apply(
        lambda row: (row['Lab_l'], row['Lab_a'], row['Lab_b']), axis=1)

    # Ensure both DataFrames are aligned and have LAB columns
    if 'lab' not in df_colorchecker.columns:
        df_colorchecker['lab'] = df_colorchecker['color'].apply(bgr_to_lab)

    # Measured LAB (from camera)
    measured_lab = np.vstack(df_colorchecker['lab'].values)
    # Reference LAB (ground truth)
    reference_lab = np.vstack(reference_value_df['lab'].values)

    # Debug: print measured and reference LAB and RGB values side by side
    print(
        "Idx | Measured LAB         | Reference LAB       | Input RGB       | "
        "Measured RGB   | Reference RGB"
    )
    for idx, (mlab, rlab) in enumerate(zip(measured_lab, reference_lab)):
        r_rgb = lab_to_bgr(rlab)
        i_rgb = df_colorchecker['color'].iloc[idx]
        print(f"{idx:2d} | {mlab} | {rlab} | {i_rgb} | {r_rgb}")

    M = fit_color_correction_matrix(measured_lab, reference_lab)

    df_colorchecker['lab_corrected'] = df_colorchecker['lab'].apply(
        lambda lab: apply_color_correction(lab, M)
    )
    # Optionally, convert corrected LAB back to RGB
    df_colorchecker['rgb_corrected'] = df_colorchecker['lab_corrected'].apply(lab_to_bgr)

    # Compute delta E (CIE76) before and after correction in a single loop
    delta_e_before = []
    delta_e_after = []
    for lab, lab_corr, ref in zip(df_colorchecker['lab'],
                                  df_colorchecker['lab_corrected'],
                                  reference_lab):
        delta_e_before.append(delta_e_cie76(lab, ref))
        delta_e_after.append(delta_e_cie76(lab_corr, ref))
    df_colorchecker['delta_e_before'] = delta_e_before
    df_colorchecker['delta_e_after'] = delta_e_after

    print("Mean delta E before correction:", np.mean(df_colorchecker['delta_e_before']))
    print("Mean delta E after correction:", np.mean(df_colorchecker['delta_e_after']))
    # print(df_colorchecker)
    return M, df_colorchecker


if __name__ == "__main__":
    # Test usage of color conversion functions
    bgr_input = (21.42603550295858, 119.26627218934911, 189.01775147928993)
    lab = bgr_to_lab(bgr_input)
    bgr_output = lab_to_bgr(lab)
    print("Input BGR:", bgr_input)
    print("Converted LAB:", lab)
    print("Converted back to BGR:", bgr_output)
