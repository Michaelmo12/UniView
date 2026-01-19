"""
Network Configuration

Network settings for TCP streaming (ports, addresses, etc.)
"""


class NetworkConfig:
    """Network configuration constants"""

    # Base port for drone TCP servers
    BASE_PORT = 15000

    # Port range: 15000-15007 (8 drones)
    # Drone 1 -> Port 15000
    # Drone 2 -> Port 15001
    # ...
    # Drone 8 -> Port 15007

    @classmethod
    def get_port(cls, drone_id):
        """
        Get TCP port for a specific drone.

        Args:
            drone_id: Drone ID (1-8)

        Returns:
            int: TCP port number
        """
        return cls.BASE_PORT + (drone_id - 1)

    @classmethod
    def get_port_range(cls, num_drones=8):
        """
        Get the full port range as a string.

        Args:
            num_drones: Number of drones (default: 8)

        Returns:
            str: Port range string (e.g., "15000-15007")
        """
        start_port = cls.BASE_PORT
        end_port = cls.BASE_PORT + num_drones - 1
        return f"{start_port}-{end_port}"
