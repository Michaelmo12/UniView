import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import cv2
import base64
import struct


def load_calibration(drone_id, frame_num, dataset_path):
    """
    Load complete calibration data for a specific drone and frame.

    Args:
        drone_id: Drone ID (1-8, matches MATRIX dataset naming)
        frame_num: Frame number (0-999)
        dataset_path: Path to MATRIX_30x30 dataset root

    Returns:
        tuple: (K, R, t, dist)
            - K: Camera intrinsic matrix (3×3 numpy array, float32)
            - R: Rotation matrix (3×3 numpy array, float32)
            - t: Translation vector (3×1 numpy array, float32)
            - dist: Distortion coefficients (5, numpy array, float32)

        Returns (None, None, None, None) if loading fails.

    """
    dataset_root = Path(dataset_path)

    try:

        K, dist = load_intrinsic(drone_id, frame_num, dataset_root)

        R, t = load_extrinsic(drone_id, frame_num, dataset_root)

        # Validate all loaded successfully
        if K is None or R is None or t is None or dist is None:
            return None, None, None, None

        return K, R, t, dist

    except Exception as e:
        print(f"Warning: Failed to load calibration for drone {drone_id}, frame {frame_num}: {e}")
        return None, None, None, None


def load_intrinsic_text(drone_id, frame_num, dataset_root):
    """
    Load raw text intrinsic data from XML file.
    Useful for network transmission - returns raw text strings.

    Args:
        drone_id: Drone ID (1-8)
        frame_num: Frame number (0-999)
        dataset_root: Path object to dataset root

    Returns:
        tuple: (K_text, dist_text)
            - K_text: Camera matrix as text string (or None)
            - dist_text: Distortion coefficients as text string (or None)
    """
    filename = f"intr_Drone{drone_id}_{frame_num:04d}.xml"
    filepath = dataset_root / "calibrations" / "intrinsic" / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Intrinsic calibration file not found: {filepath}")

    tree = ET.parse(filepath)
    root = tree.getroot()

    # Extract camera matrix text
    K_text = None
    camera_matrix_elem = root.find('.//camera_matrix')
    if camera_matrix_elem is not None:
        data_elem = camera_matrix_elem.find('.//data')
        if data_elem is not None:
            K_text = data_elem.text.strip()

    # Extract distortion coefficients text
    dist_text = None
    distortion_elem = root.find('.//distortion_coefficients')
    if distortion_elem is not None:
        data_elem = distortion_elem.find('.//data')
        if data_elem is not None:
            dist_text = data_elem.text.strip()

    return K_text, dist_text


def decode_intrinsic_text(K_text, dist_text):
    """
    Decode intrinsic parameters from text strings to numpy arrays.

    Args:
        K_text: Camera matrix as text string (or None)
        dist_text: Distortion coefficients as text string (or None)

    Returns:
        tuple: (K, dist)
            - K: Camera matrix (3×3, float32) or None
            - dist: Distortion coefficients (5, float32)
    """
    # Decode camera matrix
    if K_text is not None:
        K = decode_camera_matrix(K_text)
    else:
        K = None

    # Decode distortion coefficients
    if dist_text is not None:
        dist = decode_distortion_coefficients(dist_text)
    else:
        dist = np.zeros(5, dtype=np.float32)

    return K, dist


def load_intrinsic(drone_id, frame_num, dataset_root):
    """
    Load intrinsic camera parameters from XML file.

    Args:
        drone_id: Drone ID (1-8)
        frame_num: Frame number (0-999)
        dataset_root: Path object to dataset root

    Returns:
        tuple: (K, dist)
            - K: Camera matrix (3×3, float32)
            - dist: Distortion coefficients (5, float32)
    """
    # Load text data from file
    K_text, dist_text = load_intrinsic_text(drone_id, frame_num, dataset_root)

    # Decode to numpy arrays
    return decode_intrinsic_text(K_text, dist_text)

def decode_rvec_binary(rvec_binary):
    """
    Decode rotation vector from binary data (24 bytes).

    Args:
        rvec_binary: 24 bytes containing 3 doubles (rotation vector)

    Returns:
        np.ndarray: Rotation vector as float64 column vector (3×1)
    """
    rvec_values = struct.unpack('ddd', rvec_binary)
    return np.array(rvec_values, dtype=np.float64).reshape(3, 1)


def decode_tvec_binary(tvec_binary):
    """
    Decode translation vector from binary data (24 bytes).

    Args:
        tvec_binary: 24 bytes containing 3 doubles (translation vector)

    Returns:
        np.ndarray: Translation vector as float64 column vector (3×1)
    """
    tvec_values = struct.unpack('ddd', tvec_binary)
    return np.array(tvec_values, dtype=np.float64).reshape(3, 1)


def decode_rotation_matrix(rvec):
    """
    Convert rotation vector to rotation matrix using Rodrigues formula.

    Args:
        rvec: Rotation vector (3×1 numpy array, float64)

    Returns:
        np.ndarray: Rotation matrix (3×3, float32)
    """
    R, _ = cv2.Rodrigues(rvec)
    return R.astype(np.float32)


def decode_translation_vector(tvec):
    """
    Convert translation vector to float32.

    Args:
        tvec: Translation vector (3×1 numpy array, float64)

    Returns:
        np.ndarray: Translation vector (3×1, float32)
    """
    return tvec.astype(np.float32)


def decode_camera_matrix(matrix_text):
    """
    Decode camera matrix from text string.

    Args:
        matrix_text: Space-separated string of 9 float values

    Returns:
        np.ndarray: Camera matrix (3×3, float32)
    """
    values = [float(x) for x in matrix_text.split()]
    return np.array(values, dtype=np.float32).reshape(3, 3)


def decode_distortion_coefficients(dist_text):
    """
    Decode distortion coefficients from text string.

    Args:
        dist_text: Space-separated string of distortion coefficient values

    Returns:
        np.ndarray: Distortion coefficients (5, float32)
    """
    values = [float(x) for x in dist_text.split()]
    # Ensure exactly 5 coefficients (k1, k2, p1, p2, k3)
    if len(values) >= 5:
        return np.array(values[:5], dtype=np.float32)
    else:
        # Pad with zeros if needed
        dist = np.zeros(5, dtype=np.float32)
        dist[:len(values)] = values
        return dist


def load_extrinsic_binary(drone_id, frame_num, dataset_root):
    """
    Load raw binary extrinsic data from XML file.
    Useful for network transmission - returns raw 24-byte blobs.

    Args:
        drone_id: Drone ID (1-8)
        frame_num: Frame number (0-999)
        dataset_root: Path object to dataset root

    Returns:
        tuple: (rvec_binary, tvec_binary)
            - rvec_binary: 24 bytes (rotation vector)
            - tvec_binary: 24 bytes (translation vector)
    """
    filename = f"extr_Drone{drone_id}_{frame_num:04d}.xml"
    filepath = dataset_root / "calibrations" / "extrinsic" / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Extrinsic calibration file not found: {filepath}")

    tree = ET.parse(filepath)
    root = tree.getroot()

    # Extract rvec binary
    rvec_elem = root.find('.//rvec')
    if rvec_elem is not None:
        data_elem = rvec_elem.find('.//data')
        if data_elem is not None and data_elem.get('type_id') == 'binary':
            binary_data = base64.b64decode(data_elem.text.strip())
            rvec_binary = binary_data[-24:]  # Last 24 bytes
        else:
            raise ValueError("rvec is not in binary format")
    else:
        raise ValueError("rvec element not found in XML")

    # Extract tvec binary
    tvec_elem = root.find('.//tvec')
    if tvec_elem is not None:
        data_elem = tvec_elem.find('.//data')
        if data_elem is not None and data_elem.get('type_id') == 'binary':
            binary_data = base64.b64decode(data_elem.text.strip())
            tvec_binary = binary_data[-24:]  # Last 24 bytes
        else:
            raise ValueError("tvec is not in binary format")
    else:
        raise ValueError("tvec element not found in XML")

    return rvec_binary, tvec_binary


def load_extrinsic(drone_id, frame_num, dataset_root):
    """
    Load extrinsic camera parameters from XML file.

    IMPORTANT: The MATRIX dataset stores rvec and tvec as binary base64-encoded data.

    Args:
        drone_id: Drone ID (1-8)
        frame_num: Frame number (0-999)
        dataset_root: Path object to dataset root

    Returns:
        tuple: (R, t)
            - R: Rotation matrix (3×3, float32) converted from rvec
            - t: Translation vector (3×1, float32)
    """
    # Load binary data from file
    rvec_binary, tvec_binary = load_extrinsic_binary(drone_id, frame_num, dataset_root)

    # Decode using helper functions
    rvec = decode_rvec_binary(rvec_binary)
    tvec = decode_tvec_binary(tvec_binary)
    R = decode_rotation_matrix(rvec)
    t = decode_translation_vector(tvec)

    return R, t


if __name__ == "__main__":
    """
    Debug script to test calibration loading.
    Run: python calibration_loader.py
    """
    # Test configuration
    DATASET_PATH = Path(__file__).parent.parent.parent / "MATRIX_30x30" / "MATRIX_30x30"
    TEST_FRAME_NUM = 0
    DRONES_TO_COMPARE = [1, 2, 3, 4]  # Compare first 4 drones

    print("=" * 80)
    print("Camera Calibration Loader - Multi-Drone Comparison Mode")
    print("=" * 80)
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Frame number: {TEST_FRAME_NUM}")
    print(f"Comparing drones: {DRONES_TO_COMPARE}")
    print()

    # Store calibrations for comparison
    calibrations = {}

    # Load calibrations for each drone
    for drone_id in DRONES_TO_COMPARE:
        print(f"Loading calibration for Drone {drone_id}...")
        K, R, t, dist = load_calibration(drone_id, TEST_FRAME_NUM, DATASET_PATH)

        if K is not None:
            calibrations[drone_id] = {'K': K, 'R': R, 't': t, 'dist': dist}
            print(f"  ✓ Drone {drone_id} loaded successfully")
        else:
            print(f"  ✗ Drone {drone_id} failed to load")

    print()
    print("=" * 80)
    print("CALIBRATION COMPARISON")
    print("=" * 80)
    print()

    # Display intrinsic matrices comparison
    print("INTRINSIC MATRICES (K) - Camera internal parameters")
    print("-" * 80)
    for drone_id in sorted(calibrations.keys()):
        print(f"\nDrone {drone_id}:")
        K = calibrations[drone_id]['K']
        print(f"  Focal length (fx, fy): ({K[0,0]:.2f}, {K[1,1]:.2f})")
        print(f"  Principal point (cx, cy): ({K[0,2]:.2f}, {K[1,2]:.2f})")
        print(f"  Full matrix:\n{K}")

    print()
    print("-" * 80)
    print("ROTATION MATRICES (R) - Camera orientation")
    print("-" * 80)
    for drone_id in sorted(calibrations.keys()):
        print(f"\nDrone {drone_id}:")
        print(calibrations[drone_id]['R'])

    print()
    print("-" * 80)
    print("TRANSLATION VECTORS (t) - Camera position")
    print("-" * 80)
    for drone_id in sorted(calibrations.keys()):
        t = calibrations[drone_id]['t']
        print(f"Drone {drone_id}: x={t[0,0]:8.2f}, y={t[1,0]:8.2f}, z={t[2,0]:8.2f}")

    print()
    print("-" * 80)
    print("DISTORTION COEFFICIENTS (dist) - Lens distortion")
    print("-" * 80)
    print("Format: [k1, k2, p1, p2, k3]")
    for drone_id in sorted(calibrations.keys()):
        dist = calibrations[drone_id]['dist']
        print(f"Drone {drone_id}: {dist}")

    print()
    print("=" * 80)
    print(f"Successfully compared {len(calibrations)}/{len(DRONES_TO_COMPARE)} drones")
    print("=" * 80)

    # DEBUG: Show binary data breakdown for first drone
    print()
    print("=" * 80)
    print("DEBUG: Binary Data Analysis (Drone 1, Frame 0)")
    print("=" * 80)

    debug_drone_id = 1
    debug_frame = 0
    extr_filename = f"extr_Drone{debug_drone_id}_{debug_frame:04d}.xml"
    extr_filepath = DATASET_PATH / "calibrations" / "extrinsic" / extr_filename

    if extr_filepath.exists():
        tree = ET.parse(extr_filepath)
        root = tree.getroot()

        # Analyze rvec binary data
        rvec_elem = root.find('.//rvec')
        if rvec_elem is not None:
            data_elem = rvec_elem.find('.//data')
            if data_elem is not None and data_elem.get('type_id') == 'binary':
                binary_data = base64.b64decode(data_elem.text.strip())

                print(f"\nRVEC Binary Analysis:")
                print(f"  Total bytes: {len(binary_data)}")
                print(f"  Binary data (hex): {binary_data.hex()}")
                print()

                # Try different offsets
                print("  Trying different offsets:")
                for offset in [0, 8, 16, 24, 32]:
                    if offset + 24 <= len(binary_data):
                        try:
                            vals = struct.unpack('ddd', binary_data[offset:offset+24])
                            print(f"    Offset {offset:2d}: {vals}")
                        except:
                            print(f"    Offset {offset:2d}: Failed to unpack")

                # Show the last 24 bytes (what we actually use)
                vals_end = struct.unpack('ddd', binary_data[-24:])
                print(f"\n  Last 24 bytes (USED): {vals_end}")
                print(f"  Last 24 bytes (hex): {binary_data[-24:].hex()}")

                # Show Rodrigues conversion
                print("\n  RODRIGUES CONVERSION DEMONSTRATION:")
                rvec_demo = np.array(vals_end, dtype=np.float64).reshape(3, 1)
                R_demo, _ = cv2.Rodrigues(rvec_demo)
                print(f"  Input: Rotation Vector (3 numbers, compact storage)")
                print(f"    rvec = {vals_end}")
                print(f"\n  Output: Rotation Matrix (3×3, ready for calculations)")
                print(f"    R =")
                for row in R_demo:
                    print(f"      [{row[0]:8.5f}, {row[1]:8.5f}, {row[2]:8.5f}]")
                print(f"\n  Why convert? Matrices are easy to apply to 3D points:")
                print(f"    rotated_point = R @ original_point")

        # Analyze tvec binary data
        tvec_elem = root.find('.//tvec')
        if tvec_elem is not None:
            data_elem = tvec_elem.find('.//data')
            if data_elem is not None and data_elem.get('type_id') == 'binary':
                binary_data = base64.b64decode(data_elem.text.strip())

                print(f"\n\nTVEC Binary Analysis:")
                print(f"  Total bytes: {len(binary_data)}")
                print(f"  Binary data (hex): {binary_data.hex()}")
                print()

                # Try different offsets
                print("  Trying different offsets:")
                for offset in [0, 8, 16, 24, 32]:
                    if offset + 24 <= len(binary_data):
                        try:
                            vals = struct.unpack('ddd', binary_data[offset:offset+24])
                            print(f"    Offset {offset:2d}: {vals}")
                        except:
                            print(f"    Offset {offset:2d}: Failed to unpack")

                # Show the last 24 bytes (what we actually use)
                vals_end = struct.unpack('ddd', binary_data[-24:])
                print(f"\n  Last 24 bytes (USED): {vals_end}")
                print(f"  Last 24 bytes (hex): {binary_data[-24:].hex()}")

    print()
    print("=" * 80)
