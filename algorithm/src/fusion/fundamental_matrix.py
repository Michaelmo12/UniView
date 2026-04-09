import numpy as np


def compute_fundamental_matrix(P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """
    Calculates F from two projection matrices P1 and P2.

    Logic Flow:
    1. Find where Camera 1 is located in the 3D world (C1).
    2. Find where Camera 1 appears inside Camera 2's image (Epipole e2).
    3. Construct F to map pixels from Cam 1 to lines in Cam 2 passing through e2.
    4. Clean up mathematical noise (Rank-2 constraint).

    Args:
        P1: Projection matrix of the first drone (3x4).
        P2: Projection matrix of the second drone (3x4).

    Returns:
        F: The 3x3 Fundamental Matrix.

    Note: Camera center can also be computed from R, t as C = -R.T @ t
    (see CameraCalibration.camera_center in ingestion/models.py).
    We use SVD here to avoid dependency on the ingestion module.
    """
    assert P1.shape == (3, 4), f"P1 must be (3, 4), got {P1.shape}"
    assert P2.shape == (3, 4), f"P2 must be (3, 4), got {P2.shape}"

    # Step 1: Find Camera 1's physical location (C1)
    # Formula: P1 @ C1 = 0
    # We are looking for the "Camera Center".
    # Mathematically, this is the only 3D point that projects to "0" (disappears)
    # because you cannot take a picture of the camera's own lens center.
    # We use SVD to find this "Null Space".
    # svd breaks P1 into U, S, Vt such that P1 = U @ S @ Vt
    _, _, Vt = np.linalg.svd(P1)
    C1 = Vt[-1, :]  # The last row of V^T is the solution C1 where P1 @ C1 = 0

    # Step 2: Find the Epipole in Image 2 (e2)
    # Formula: e2 = P2 @ C1
    # The epipole is the projection of Camera 1's center onto Camera 2's image plane.
    # It answers: "If Drone 2 took a picture of Drone 1, where would it be?"
    e2 = P2 @ C1  # Shape: (3,)

    # Step 3: Create a helper matrix for drawing lines
    # Formula: Line = e2 x Point  =>  Matrix [e2]_x
    # We need to compute a "Cross Product" to draw a line between the epipole and a point.
    # Computers prefer matrix multiplication, so we convert vector e2 into a special
    # "Skew-Symmetric" matrix. Multiplying by this matrix is the same as doing a cross product.
    e2_cross = np.array(
        [[0, -e2[2], e2[1]],
         [e2[2], 0, -e2[0]],
         [-e2[1], e2[0], 0]], dtype=np.float64
    )

    # Step 4: Compute the raw Fundamental Matrix F
    # Formula: F = [e2]_x @ P2 @ P1^+
    # The logic is a chain reaction:
    # 1. Take a pixel from Image 1.
    # 2. 'P1_pinv' (P1^+) sends it back into 3D space (Pseudo-Inverse).
    # 3. 'P2' projects that 3D point onto Image 2.
    # 4. 'e2_cross' ([e2]_x) connects that point to the epipole to form a line.
    P1_pinv = np.linalg.pinv(P1)  # Shape: (4, 3)
    F = e2_cross @ P2 @ P1_pinv  # Shape: (3, 3)

    # Step 5: Clean up noise (Enforce Rank-2)
    # Formula: F_clean = U @ diag(s1, s2, 0) @ Vt
    # Due to computer rounding erro rs, the calculated F might be slightly "broken" (Rank 3).
    # A valid F matrix must have Rank 2 (it maps points to lines, not points to points).
    # We use SVD to find the smallest noise component (sigma 3) and delete it.
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0.0  # Zero out the smallest singular value (the noise)
    F_rank2 = U @ np.diag(S) @ Vt  # Rebuild the perfect Rank-2 matrix,
    # we want f to give us lines that pass through the epipole, just like in Ax =0, we want to remove the noise that makes it not pass through the epipole.

    return F_rank2


def compute_fundamental_matrix_batch(
    projection_matrices: dict[int, np.ndarray],
) -> dict[tuple[int, int], np.ndarray]:
    """
    Compute fundamental matrices for all camera pairs.

    Args:
        projection_matrices: {drone_id: projection_matrix (3, 4)}

    Returns:
        {(drone_id_i, drone_id_j): F_ij} for all i < j
        F_ij maps points from camera i to epipolar lines in camera j
    """
    result = {}
    drone_ids = sorted(projection_matrices.keys())

    for i in range(len(drone_ids)):
        for j in range(i + 1, len(drone_ids)):
            drone_id_i = drone_ids[i]
            drone_id_j = drone_ids[j]

            P_i = projection_matrices[drone_id_i]
            P_j = projection_matrices[drone_id_j]

            F_ij = compute_fundamental_matrix(P_i, P_j)

            result[(drone_id_i, drone_id_j)] = F_ij

    return result

