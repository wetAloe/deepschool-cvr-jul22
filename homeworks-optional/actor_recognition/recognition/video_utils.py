from pathlib import Path
import numpy as np
from subprocess import Popen, PIPE
import ffmpeg
import logging
import typing as t
from .types import RGBImage


def read_frame(process1: Popen, width: int, height: int) -> RGBImage:
    logging.debug('Reading frame')

    # Note: RGB24 == 3 bytes per pixel.
    frame_size = width * height * 3
    in_bytes = process1.stdout.read(frame_size)
    if len(in_bytes) == 0:
        frame = None
    else:
        assert len(in_bytes) == frame_size
        frame = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([height, width, 3])
        )
    return frame


def write_frame(process2: Popen, frame: RGBImage) -> None:
    logging.debug('Writing frame')
    process2.stdin.write(
        frame
        .astype(np.uint8)
        .tobytes()
    )


def get_video_size(filename: Path) -> t.Tuple[int, int]:
    logging.info('Getting video size for {!r}'.format(filename))
    probe = ffmpeg.probe(str(filename))
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    return width, height


def start_ffmpeg_process_in(in_filename: Path) -> Popen:
    logging.info('Starting ffmpeg process for reading')
    args = (
        ffmpeg
        .input(str(in_filename))
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .compile()
    )
    return Popen(args, stdout=PIPE)


def start_ffmpeg_process_out(out_filename: Path, width: int, height: int) -> Popen:
    logging.info('Starting ffmpeg process for writing')
    args = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
        .output(str(out_filename), pix_fmt='yuv420p')
        .overwrite_output()
        .compile()
    )
    return Popen(args, stdin=PIPE)
