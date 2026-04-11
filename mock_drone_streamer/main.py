"""
ENet Drone Streamer — Single Drone Entry Point

Starts a single ENet drone streamer for the specified drone ID.

Usage:
    python main.py --drone-id 1 --dataset /path/to/MATRIX
    python main.py --drone-id 3 --dataset /path/to/MATRIX --fps 5 --jpeg-quality 70

Run with --help for full option list.
"""

import argparse
import logging
import sys
from pathlib import Path

from mock_drone_streamer.config.config import StreamerConfig
from mock_drone_streamer.src.streamer import ENetStreamer

DEFAULT_DATASET = str(Path(__file__).parent.parent / "MATRIX_30x30" / "MATRIX_30x30")


def _setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ENet drone streamer — streams MATRIX dataset frames over ENet UDP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--drone-id",
        type=int,
        default=1,
        metavar="N",
        help="Drone ID to stream (1-8). Port = base_port + N - 1.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        metavar="PATH",
        help="Root path to MATRIX dataset directory.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        metavar="ADDR",
        help="ENet bind address.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        metavar="PORT",
        help="Override listening port. Default: base_port + drone_id - 1 (16000 + N - 1).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        metavar="FPS",
        help="Target frame send rate.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        metavar="Q",
        help="JPEG compression quality (0-100).",
    )
    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="Stop when dataset frames are exhausted (default: loop forever).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG log level.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _setup_logging(logging.DEBUG if args.debug else logging.INFO)

    logger = logging.getLogger(__name__)

    config = StreamerConfig(
        dataset_path=args.dataset,
        drone_id=args.drone_id,
        host=args.host,
        base_port=(args.port - (args.drone_id - 1)) if args.port is not None else 16000,
        fps=args.fps,
        jpeg_quality=args.jpeg_quality,
        loop=not args.no_loop,
    )

    # Override derived port if --port was explicitly provided
    if args.port is not None:
        # Patch config so config.port returns the explicit value
        config.base_port = args.port - (args.drone_id - 1)

    logger.info(
        "Starting ENet streamer: Drone %d on port %d (dataset=%s, fps=%.1f)",
        config.drone_id,
        config.port,
        config.dataset_path,
        config.fps,
    )

    streamer = ENetStreamer(config)
    streamer.run()


if __name__ == "__main__":
    main()
