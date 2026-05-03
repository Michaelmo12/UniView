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

from mock_drone_streamer.config.config import StreamerConfig
from mock_drone_streamer.src.dataset_loader import DatasetLoader
from mock_drone_streamer.src.packet_builder import PacketBuilder

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
            # Base path to dataset containing subfolders for each drone (e.g. "dataset/drone_1")
            dataset_path=config.dataset_path,
            # ID of the drone (1-based index, used to select dataset subfolder and port)
            drone_id=config.drone_id,
            # JPEG quality for encoding frames (0-100, higher is better quality and larger size)
            jpeg_quality=config.jpeg_quality,
        )
        self._builder = PacketBuilder()
        # List of currently connected ENet peers
        self._peers: list = []
        # Total frames in the dataset for this drone
        self._frame_count = self._loader.get_frame_count()
        # Index of the next frame to send
        self._current_frame_idx = 0
        # Total frames sent (for stats)
        self._frames_sent = 0
        # Total bytes sent (for stats)
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
            # Bind address: IP as bytes (C lib needs bytes, not str) + port number
            enet.Address(self.config.host.encode(), port),
            # Maximum number of peers (clients) that can connect
            peerCount=10,
            # Only 1 channel needed — we send one type of data (frame packets) on channel 0
            channelLimit=1,
            # 0 = unlimited incoming bandwidth (no throttle)
            incomingBandwidth=0,
            # 0 = unlimited outgoing bandwidth (no throttle)
            outgoingBandwidth=0,
        )

        # how long to wait between frames in seconds (e.g. 30fps → 0.0333s)
        frame_interval = 1.0 / self.config.fps
        # Subtract interval so first frame sends immediately (not after one full wait)
        last_send_time = time.monotonic() - frame_interval

        try:
            # Infinite loop until keyboard interrupt or dataset exhausted (if loop=False)
            while True:
                # Wait up to 10ms for a network event (connect/disconnect/receive), returns None if nothing happened
                # 10ms keeps loop responsive without burning 100% CPU
                event = host.service(10)
                # Process the event (if any) — updates peer list on connect/disconnect
                self._handle_event(event)

                # Send next frame on timer tick for the configured FPS
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
            # Force all buffered outgoing packets (disconnect messages) to actually be sent before exit
            host.flush()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _handle_event(self, event) -> None:
        # host.service() always returns — even when nothing happened (returns None)
        # must check before accessing event.type to avoid crash
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
            # Rebuild list without the disconnected peer
            # (safer than .remove() — never crashes and handles duplicates)
            updated_peers = []
            for p in self._peers:
                if p != event.peer:
                    updated_peers.append(p)
            self._peers = updated_peers

        # We don't expect inbound data — streamer is send-only
        # log at debug level (hidden by default) so it doesn't spam normal output
        elif event.type == enet.EVENT_TYPE_RECEIVE:
            logger.debug(
                "ENetStreamer: Drone %d — received unexpected data (%d bytes)",
                self.config.drone_id,
                len(event.packet.data),
            )

    def _send_frame(self) -> None:
        """Load current frame index and broadcast to all connected peers."""
        # Snapshot the index so the whole method works on the same frame
        frame_idx = self._current_frame_idx

        if not self._peers:
            # Still advance even with no clients — realistic simulation (drone keeps flying)
            # Only print every 10 frames to avoid terminal spam
            if frame_idx % 10 == 0:
                print(
                    f"[DRONE {self.config.drone_id}] Processing frame {frame_idx:04d} (no clients connected)"
                )
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
            # Skip broken frame and continue — realistic simulation
            self._advance_frame()
            return

        # Timestamp at moment of sending (not loading) for accurate sync
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

        # PACKET_FLAG_RELIABLE = TCP-like guarantees over UDP (resend on loss, in-order, no duplicates)
        enet_packet = enet.Packet(packet_data, enet.PACKET_FLAG_RELIABLE)

        # list() creates a copy — safe to iterate even if a peer disconnects mid-loop
        for peer in list(self._peers):
            try:
                # 0 = channel number (we only have channel 0)
                peer.send(0, enet_packet)
            except Exception as exc:
                logger.warning(
                    "ENetStreamer: Drone %d — failed to send to peer: %s",
                    self.config.drone_id,
                    exc,
                )

        self._frames_sent += 1
        self._total_bytes_sent += len(packet_data)

        # Print stats every 100 frames — enough feedback without spamming
        if frame_idx % 100 == 0:
            avg_kb = (
                (self._total_bytes_sent / self._frames_sent / 1024.0)
                if self._frames_sent
                else 0
            )
            print(
                f"[DRONE {self.config.drone_id}] Frame {frame_idx:04d}, "
                f"{len(packet_data) / 1024:.1f} KB (avg: {avg_kb:.1f} KB), "
                f"{len(self._peers)} client(s)"
            )

        self._advance_frame()

    def _advance_frame(self) -> None:
        """Advance the frame index, looping if configured."""
        self._current_frame_idx += 1
        if self.config.max_frames > 0 and self._current_frame_idx >= self.config.max_frames:
            logger.info(
                "ENetStreamer: Drone %d — max_frames=%d reached, stopping.",
                self.config.drone_id,
                self.config.max_frames,
            )
            raise StopIteration("max_frames reached")
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
                # Uncaught — bubbles to finally block for cleanup, then exits
                raise StopIteration("Dataset exhausted and loop=False")
