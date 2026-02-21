"""
PersonClusterer

Clusters triangulated 3D points into unique persons using DBSCAN and preserves
single-view detections (detections seen by only one camera).

Key Class:
- PersonClusterer: DBSCAN clustering + single-view preservation -> list[Person3D]
"""

import logging
import numpy as np
from sklearn.cluster import DBSCAN

from algorithm.config.settings import ReconstructionConfig
from algorithm.reconstruction.models import Point3D, Person3D

logger = logging.getLogger(__name__)


class PersonClusterer:
    """
    Clusters triangulated 3D points into unique persons.

    Uses DBSCAN to group nearby triangulated points (same person observed in
    multiple views). Preserves single-view detections as separate Person3D
    instances with is_triangulated=False.

    Constructor takes ReconstructionConfig with DBSCAN parameters.
    """

    def __init__(self, config: ReconstructionConfig):
        """
        Initialize clusterer with configuration.

        Args:
            config: ReconstructionConfig with dbscan_eps and dbscan_min_samples
        """
        self.config = config

    def cluster_persons(
        self,
        triangulated_points: list[Point3D],
        unmatched_detections: list[tuple[int, int]]
    ) -> list[Person3D]:
        """
        Cluster triangulated points and preserve single-view detections.

        Flow:
        1. DBSCAN clusters triangulated points (matched detections)
        2. Each cluster -> Person3D with centroid position, is_triangulated=True
        3. Noise points (isolated triangulated) -> Person3D with individual position
        4. Unmatched detections (single-view) -> Person3D with is_triangulated=False
        5. Assign sequential person_id across all persons

        Args:
            triangulated_points: List of Point3D from successful triangulations
            unmatched_detections: List of (drone_id, local_id) seen by only one camera

        Returns:
            List of Person3D (clustered + noise + single-view)
        """
        persons = []

        # Handle triangulated points if any exist
        if triangulated_points:
            # Build position array for DBSCAN
            positions = np.array([p.position for p in triangulated_points])  # (N, 3)

            # Run DBSCAN
            clusterer = DBSCAN(
                eps=self.config.dbscan_eps,
                min_samples=self.config.dbscan_min_samples,
                metric='euclidean'
            )
            labels = clusterer.fit_predict(positions)

            # Group points by cluster label
            unique_labels = set(labels)

            for label in unique_labels:
                # Get points in this cluster
                mask = labels == label
                cluster_points = [p for p, m in zip(triangulated_points, mask) if m]

                if label == -1:
                    # Noise points: isolated triangulated points
                    # Create separate Person3D for each noise point
                    for point in cluster_points:
                        persons.append(Person3D(
                            person_id=-1,  # Will be reassigned later
                            position=point.position,
                            num_views=point.num_views,
                            source_detections=point.source_detections,
                            is_triangulated=True
                        ))
                else:
                    # Regular cluster: compute centroid
                    cluster_positions = [p.position for p in cluster_points]
                    centroid = np.mean(cluster_positions, axis=0)

                    # Collect all source detections from cluster
                    all_detections = []
                    total_views = 0
                    for point in cluster_points:
                        all_detections.extend(point.source_detections)
                        total_views += point.num_views

                    persons.append(Person3D(
                        person_id=-1,  # Will be reassigned later
                        position=centroid,
                        num_views=total_views,
                        source_detections=all_detections,
                        is_triangulated=True
                    ))

        # Add single-view detections (not in any match group)
        for drone_id, local_id in unmatched_detections:
            persons.append(Person3D(
                person_id=-1,  # Will be reassigned later
                position=None,  # No 3D position available (single view)
                num_views=1,
                source_detections=[(drone_id, local_id)],
                is_triangulated=False
            ))

        # Assign sequential person IDs
        for i, person in enumerate(persons):
            person.person_id = i

        logger.debug(
            "Clustered %d triangulated points + %d single-view -> %d persons",
            len(triangulated_points), len(unmatched_detections), len(persons)
        )

        return persons


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing PersonClusterer")
    logger.info("=" * 60)

    from algorithm.config.settings import settings

    clusterer = PersonClusterer(settings.reconstruction)
    logger.info("PersonClusterer created with config:")
    logger.info("  dbscan_eps: %.1f", settings.reconstruction.dbscan_eps)
    logger.info("  dbscan_min_samples: %d", settings.reconstruction.dbscan_min_samples)

    # Test with mock data
    logger.info("\nTest clustering with mock triangulated points:")

    # Create two clusters of points
    point1 = Point3D(
        position=np.array([1.0, 2.0, 0.5]),
        reprojection_error=2.0,
        num_views=2,
        source_detections=[(1, 0), (2, 0)]
    )
    point2 = Point3D(
        position=np.array([1.1, 2.1, 0.6]),  # Close to point1
        reprojection_error=2.5,
        num_views=2,
        source_detections=[(3, 1), (4, 1)]
    )
    point3 = Point3D(
        position=np.array([5.0, 6.0, 1.0]),  # Far from others
        reprojection_error=1.5,
        num_views=2,
        source_detections=[(5, 2), (6, 2)]
    )

    unmatched = [(7, 0), (8, 1)]  # Two single-view detections

    persons = clusterer.cluster_persons([point1, point2, point3], unmatched)

    logger.info("Result: %d persons", len(persons))
    for person in persons:
        logger.info("  %s", person)

    logger.info("\n" + "=" * 60)
    logger.info("PersonClusterer ready for use")
