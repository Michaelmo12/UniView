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
    # Due to computer rounding errors, the calculated F might be slightly "broken" (Rank 3).
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


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Testing Fundamental Matrix Computation")
    logger.info("=" * 60)

    # Create two synthetic projection matrices
    # P = K @ [R | t]
    K = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])

    # Camera 1: at origin, looking down +Z
    R1 = np.eye(3)
    t1 = np.array([[0.0], [0.0], [0.0]])
    P1 = K @ np.hstack([R1, t1])

    # Camera 2: translated in X, looking down +Z
    R2 = np.eye(3)
    t2 = np.array([[5.0], [0.0], [0.0]])  # 5m to the right
    P2 = K @ np.hstack([R2, t2])

    logger.info("\nCamera 1 projection matrix:")
    logger.info("P1 =\n%s", P1)

    logger.info("\nCamera 2 projection matrix:")
    logger.info("P2 =\n%s", P2)

    logger.info("\nComputing fundamental matrix F...")
    F = compute_fundamental_matrix(P1, P2)

    logger.info("\nFundamental matrix F:")
    logger.info("F =\n%s", F)

    # Check rank-2 constraint
    rank = np.linalg.matrix_rank(F)
    logger.info("\nRank of F: %d (should be 2)", rank)
    assert rank == 2, f"F must have rank 2, got {rank}"

    # Test epipolar constraint with a test point
    # Point in camera 1: (500, 300) in homogeneous coords
    x1 = np.array([500.0, 300.0, 1.0])

    # For perfect correspondence, x2^T @ F @ x1 should be ~0
    # But we'll just compute the epipolar line in camera 2
    epiline_2 = F @ x1  # Shape: (3,) - line coefficients [a, b, c]

    logger.info("\nTest epipolar constraint:")
    logger.info("  Point in camera 1: x1 = %s", x1[:2])
    logger.info("  Epipolar line in camera 2: %s", epiline_2)
    logger.info("  (Points in camera 2 must satisfy: ax + by + c = 0)")

    # Test batch computation
    logger.info("\nTesting batch computation...")
    projection_matrices = {1: P1, 2: P2}
    F_batch = compute_fundamental_matrix_batch(projection_matrices)

    logger.info("  Computed %d fundamental matrices", len(F_batch))
    for (i, j), F_ij in F_batch.items():
        logger.info("  F_%d_%d rank: %d", i, j, np.linalg.matrix_rank(F_ij))

    logger.info("\n" + "=" * 60)
    logger.info("All tests passed!")
