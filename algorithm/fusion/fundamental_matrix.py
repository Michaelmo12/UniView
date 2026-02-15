"""
Fundamental Matrix Computation

Computes fundamental matrices from projection matrix pairs for epipolar geometry.

The fundamental matrix F encodes the epipolar constraint:
    x2^T @ F @ x1 = 0

Where x1 and x2 are corresponding points in two camera views (homogeneous coords).
This constraint says: a point in camera 1 must lie on its corresponding epipolar
line in camera 2.

White-box: Standard computer vision formula with rank-2 enforcement via SVD.
"""

import numpy as np


def compute_fundamental_matrix(P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """
    Compute fundamental matrix from two projection matrices.

    The fundamental matrix F relates corresponding points in two views:
        x2^T @ F @ x1 = 0

    Where x1, x2 are homogeneous pixel coordinates in images 1 and 2.

    Algorithm (Hartley & Zisserman, "Multiple View Geometry", Section 9.2.2):
    1. Compute camera center C1 from P1 (null space of P1)
    2. Project C1 to image 2: e2 = P2 @ C1 (epipole in image 2)
    3. Compute F = [e2]_x @ P2 @ P1^+
       - [e2]_x is the skew-symmetric matrix of e2 (cross-product operator)
       - P1^+ is the pseudoinverse of P1
    4. Enforce rank-2 constraint via SVD

    Args:
        P1: First projection matrix (3, 4)
        P2: Second projection matrix (3, 4)

    Returns:
        F: Fundamental matrix (3, 3) with rank 2
    """
    assert P1.shape == (3, 4), f"P1 must be (3, 4), got {P1.shape}"
    assert P2.shape == (3, 4), f"P2 must be (3, 4), got {P2.shape}"

    # Step 1: Compute camera center C1 (null space of P1)
    # P1 @ C1 = 0, so C1 is the right null vector of P1
    # We use SVD: P1 = U @ S @ Vt, null space is last column of V
    _, _, Vt = np.linalg.svd(P1)
    C1 = Vt[-1, :]  # Last row of Vt = last column of V, shape (4,)

    # Step 2: Compute epipole in image 2
    # e2 = P2 @ C1 (project camera 1 center to image 2)
    e2 = P2 @ C1  # Shape: (3,)

    # Step 3: Construct skew-symmetric matrix [e2]_x
    # [e2]_x represents the cross product operator
    # For vector e2 = [a, b, c], [e2]_x is:
    #     [  0, -c,  b ]
    #     [  c,  0, -a ]
    #     [ -b,  a,  0 ]
    e2_cross = np.array(
        [[0, -e2[2], e2[1]], [e2[2], 0, -e2[0]], [-e2[1], e2[0], 0]], dtype=np.float64
    )

    # Step 4: Compute pseudoinverse of P1
    # P1^+ = P1^T @ (P1 @ P1^T)^(-1) for full-rank P1
    # But np.linalg.pinv is more numerically stable
    P1_pinv = np.linalg.pinv(P1)  # Shape: (4, 3)

    # Step 5: Compute F = [e2]_x @ P2 @ P1^+
    F = e2_cross @ P2 @ P1_pinv  # Shape: (3, 3)

    # Step 6: Enforce rank-2 constraint
    # F should have rank 2 (det(F) = 0)
    # We enforce this by zeroing the smallest singular value
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0.0  # Zero smallest singular value
    F_rank2 = U @ np.diag(S) @ Vt

    return F_rank2


def compute_fundamental_matrix_batch(
    projection_matrices: dict[int, np.ndarray],
) -> dict[tuple[int, int], np.ndarray]:
    """
    Compute fundamental matrices for all camera pairs.

    For N cameras, this computes N*(N-1)/2 fundamental matrices
    (one for each unique pair).

    Args:
        projection_matrices: {drone_id: projection_matrix (3, 4)}

    Returns:
        {(drone_id_i, drone_id_j): F_ij} for all i < j
        F_ij maps points from camera i to epipolar lines in camera j
    """
    result = {}
    drone_ids = sorted(projection_matrices.keys())

    # Iterate over all unique pairs (i, j) where i < j
    for i in range(len(drone_ids)):
        for j in range(i + 1, len(drone_ids)):
            drone_id_i = drone_ids[i]
            drone_id_j = drone_ids[j]

            P_i = projection_matrices[drone_id_i]
            P_j = projection_matrices[drone_id_j]

            # Compute F_ij: maps points from camera i to lines in camera j
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
