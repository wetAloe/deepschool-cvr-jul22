from pathlib import Path
import numpy as np
from time import time
import click
import logging

from recognition.face_tracker import FaceTracker
from recognition.recognizer import FaceRecognizer
from recognition.gallery import config_gallery_finder
from recognition.video_utils import get_video_size, start_ffmpeg_process_in, start_ffmpeg_process_out, \
    read_frame, write_frame


@click.command()
@click.argument('input_video', type=Path)
@click.argument('output_video', type=Path)
@click.option('-d', '--detector-path', type=Path, default=Path(__file__).parent.parent / 'models' / 'scrfd_10g_bnkps.onnx')
@click.option('-e', '--embedder-path', type=Path, default=Path(__file__).parent.parent / 'models' / 'glintr100.onnx')
@click.option('-g', '--gallery-root', type=Path, default=Path(__file__).parent.parent / 'actors')
def run(input_video, output_video, detector_path, embedder_path, gallery_root):
    logging.basicConfig(level=logging.INFO)
    recognizer = FaceRecognizer(detector_path, embedder_path)
    gallery_finder = config_gallery_finder(recognizer, gallery_root)
    tracker = FaceTracker(recognizer, gallery_finder)
    process_frame = tracker.process_frame
    start = time()
    width, height = get_video_size(input_video)
    process1 = start_ffmpeg_process_in(input_video)
    process2 = start_ffmpeg_process_out(output_video, width, height)
    while True:
        in_frame = read_frame(process1, width, height)
        if in_frame is None:
            logging.info('End of input stream')
            break

        logging.debug('Processing frame')
        out_frame = process_frame(in_frame, width, height)
        write_frame(process2, out_frame)

    logging.info('Waiting for ffmpeg process1')
    process1.wait()

    logging.info('Waiting for ffmpeg process2')
    process2.stdin.close()
    process2.wait()

    logging.info('Done')
    end = time()
    t = end - start
    logging.info(f'Processed {tracker.num_frames} frames in {t} s; {tracker.num_frames / t} fps')
    logging.info(f'Mean processing times: {np.mean(tracker.times)}')


if __name__ == '__main__':
    run()
