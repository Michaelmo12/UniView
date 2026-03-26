"""
ENetStreamer - ENet UDP Server for Drone Frame Streaming

Runs an ENet server that continuously sends binary drone frame packets
to all connected peers. Peers connect as ENet clients (see ENetReceiver).

Architecture:
    - Binds an ENet host on the configured port (base_port + drone_id - 1)
    - Services ENet events in a tight loop: CONNECT, DISCONNECT, RECEIVE
    - On each frame tick, loads the next frame and sends it reliably to all peers
    - Loops the dataset when frames are exhausted (if config.loop is True)

ENet reliability:
    All packets are sent on channel 0 with PACKET_FLAG_RELIABLE, which gives
    TCP-like delivery guarantees over UDP with lower latency than TCP.
"""

import logging
import time

import enet  # pyenet

from enet_drone_streamer.config.config import StreamerConfig
from enet_drone_streamer.src.dataset_loader import DatasetLoader
from enet_drone_streamer.src.packet_builder import PacketBuilder

logger = logging.getLogger(__name__)


class ENetStreamer:
    """
    ENet server that sends drone frame packets to connected peers.

    Args:
        config: StreamerConfig instance controlling port, fps, dataset, etc.
    """

    def __init__(self, config: StreamerConfig) -> None:
        self.config = config
        self._loader = DatasetLoader(
            dataset_path=config.dataset_path,
            drone_id=config.drone_id,
            jpeg_quality=config.jpeg_quality,
        )
        self._builder = PacketBuilder()
        self._peers: list = []
        self._frame_count = self._loader.get_frame_count()
        self._current_frame_idx = 0
        self._frames_sent = 0
        self._total_bytes_sent = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Run the ENet server loop.

        Blocks until KeyboardInterrupt.  Cleans up the ENet host on exit.
        """
        port = self.config.port
        logger.info(
            "ENetStreamer: Drone %d starting on %s:%d  (%.1f fps, %d frames, loop=%s)",
            self.config.drone_id,
            self.config.host,
            port,
            self.config.fps,
            self._frame_count,
            self.config.loop,
        )

        host = enet.Host(
            enet.Address(self.config.host.encode(), port),
            peerCount=10,
            channelLimit=1,
            incomingBandwidth=0,
            outgoingBandwidth=0,
        )

        frame_interval = 1.0 / self.config.fps
        last_send_time = time.monotonic() - frame_interval  # send immediately on start

        try:
            while True:
                # Service all pending ENet events (10 ms timeout keeps loop responsive)
                event = host.service(10)
                self._handle_event(event)

                # Send next frame on timer tick
                now = time.monotonic()
                if now - last_send_time >= frame_interval:
                    self._send_frame()
                    last_send_time = now

        except KeyboardInterrupt:
            logger.info(
                "ENetStreamer: Drone %d shutting down (sent %d frames to %d peer(s))",
                self.config.drone_id,
                self._current_frame_idx,
                len(self._peers),
            )
        finally:
            # Disconnect all peers cleanly
            for peer in list(self._peers):
                try:
                    peer.disconnect()
                except Exception:
                    pass
            host.flush()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _handle_event(self, event) -> None:
        if event is None:
            return

        if event.type == enet.EVENT_TYPE_CONNECT:
            logger.info(
                "ENetStreamer: Drone %d — peer connected from %s",
                self.config.drone_id,
                event.peer.address,
            )
            self._peers.append(event.peer)

        elif event.type == enet.EVENT_TYPE_DISCONNECT:
            logger.info(
                "ENetStreamer: Drone %d — peer disconnected",
                self.config.drone_id,
            )
            self._peers = [p for p in self._peers if p != event.peer]

        elif event.type == enet.EVENT_TYPE_RECEIVE:
            # We don't expect inbound data but log it if received
            logger.debug(
                "ENetStreamer: Drone %d — received unexpected data (%d bytes)",
                self.config.drone_id,
                len(event.packet.data),
            )

    def _send_frame(self) -> None:
        """Load current frame index and broadcast to all connected peers."""
        frame_idx = self._current_frame_idx

        if not self._peers:
            if frame_idx % 10 == 0:
                print(f"[DRONE {self.config.drone_id}] Processing frame {frame_idx:04d} (no clients connected)")
            self._advance_frame()
            return

        try:
            jpeg_bytes, K, R, t, dist = self._loader.load_frame(frame_idx)
        except Exception as exc:
            logger.error(
                "ENetStreamer: Drone %d — failed to load frame %d: %s",
                self.config.drone_id,
                frame_idx,
                exc,
            )
            self._advance_frame()
            return

        timestamp_ns = time.time_ns()
        packet_data = self._builder.build_packet(
            drone_id=self.config.drone_id,
            frame_num=frame_idx,
            timestamp_ns=timestamp_ns,
            jpeg_bytes=jpeg_bytes,
            K=K,
            R=R,
            t=t,
            dist=dist,
        )

        enet_packet = enet.Packet(packet_data, enet.PACKET_FLAG_RELIABLE)

        for peer in list(self._peers):
            try:
                peer.send(0, enet_packet)
            except Exception as exc:
                logger.warning(
                    "ENetStreamer: Drone %d — failed to send to peer: %s",
                    self.config.drone_id,
                    exc,
                )

        self._frames_sent += 1
        self._total_bytes_sent += len(packet_data)

        if frame_idx % 100 == 0:
            avg_kb = (self._total_bytes_sent / self._frames_sent / 1024.0) if self._frames_sent else 0
            print(
                f"[DRONE {self.config.drone_id}] Frame {frame_idx:04d}, "
                f"{len(packet_data) / 1024:.1f} KB (avg: {avg_kb:.1f} KB), "
                f"{len(self._peers)} client(s)"
            )

        self._advance_frame()

    def _advance_frame(self) -> None:
        """Advance the frame index, looping if configured."""
        self._current_frame_idx += 1
        if self._current_frame_idx >= self._frame_count:
            if self.config.loop:
                logger.debug(
                    "ENetStreamer: Drone %d — looping dataset back to frame 0",
                    self.config.drone_id,
                )
                self._current_frame_idx = 0
            else:
                logger.info(
                    "ENetStreamer: Drone %d — dataset exhausted, stopping.",
                    self.config.drone_id,
                )
                raise StopIteration("Dataset exhausted and loop=False")
