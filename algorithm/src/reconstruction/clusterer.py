import logging
import numpy as np

from src.config.settings import ReconstructionConfig
from src.reconstruction.dbscan import WhiteBoxDBSCAN
from src.reconstruction.models import Point3D, Person3D

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
        unmatched_detections: list[tuple[int, int]],  # single views
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
        # The output list — all Person3D objects
        persons = []
        person_id_counter = 0

        # Handle triangulated points if any exist
        if triangulated_points:
            # DBSCAN expects array of points not objects
            position_rows = []
            for point in triangulated_points:
                position_rows.append(point.position)
            positions = np.array(position_rows)  # (N, 3)

            # Run DBSCAN expects array of points not objects
            clusterer = WhiteBoxDBSCAN(
                eps=self.config.dbscan_eps,
                min_samples=self.config.dbscan_min_samples,
            )
            """runs the full algorithm in one call and returns a label array the same length as positions:

            positions:  [ [1.0, 2.0, 0.5],  [1.1, 2.1, 0.6],  [5.0, 6.0, 1.0] ]
            labels:     [       0        ,        0        ,        -1        ]

            -1 means noise (point3), 0 means cluster 0 (point1 and point2)
            """
            labels = clusterer.fit_predict(positions)

            # Group points by cluster label
            unique_labels = set(labels)

            for label in unique_labels:
                # Get points in this cluster
                """labels = [0, 0, -1, 1, 1, 0]
                label  = 0

                mask   = [True, True, False, False, False, True]
                """
                mask = labels == label
                cluster_points = []
                for point, is_in_cluster in zip(triangulated_points, mask):
                    if is_in_cluster:
                        cluster_points.append(point)

                if label == -1:
                    # Noise points: isolated triangulated points
                    # Create separate Person3D for each noise point
                    for point in cluster_points:
                        persons.append(
                            Person3D(
                                person_id=person_id_counter,
                                position=point.position,
                                num_views=2,  # each noise point is always one pairwise triangulation
                                source_detections=point.source_detections,
                                is_triangulated=True,
                            )
                        )
                        person_id_counter += 1
                else:
                    # Regular cluster: compute centroid
                    cluster_positions = []
                    # get all positions of the cluster
                    for point in cluster_points:
                        cluster_positions.append(point.position)

                    # compute avg [avg_x, avg_y, avg_z]
                    centroid = np.mean(cluster_positions, axis=0)

                    # Collects all unique (drone_id, local_id)
                    unique_detections = set()
                    for point in cluster_points:
                        for detection in point.source_detections:
                            unique_detections.add(detection)
                    
                    # SET->LIST for Person3D object
                    all_detections = list(unique_detections)
                    # how many cameras
                    total_views = len(
                        all_detections
                    )  # unique cameras that saw this person

                    persons.append(
                        Person3D(
                            person_id=person_id_counter,
                            position=centroid,
                            num_views=total_views,
                            source_detections=all_detections,
                            is_triangulated=True,
                        )
                    )
                    person_id_counter += 1

        # Add single-view detections (not in any match group)
        for drone_id, local_id in unmatched_detections:
            persons.append(
                Person3D(
                    person_id=person_id_counter,
                    position=None,  # No 3D position available (single view)
                    num_views=1,
                    source_detections=[(drone_id, local_id)],
                    is_triangulated=False,
                )
            )
            person_id_counter += 1

        logger.debug(
            "Clustered %d triangulated points + %d single-view -> %d persons",
            len(triangulated_points),
            len(unmatched_detections),
            len(persons),
        )

        return persons
