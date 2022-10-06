import argparse
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import numpy as np
import requests

from constants import PROJECT_PATH
from utils import init_logging

CHUNK_SIZE = 100


def do_request(endpoint_url, upload_file):
    return requests.post(endpoint_url, files=upload_file)


def run_stress(images_dir: str, endpoint_url: str, threads: int, n_repeats: int):
    img_names = os.listdir(images_dir)
    do_request_to_endpoint = partial(do_request, endpoint_url)
    perfomances = []
    with ThreadPoolExecutor(max_workers=threads) as executor:
        for _ in range(n_repeats):
            imgs_paths = [os.path.join(images_dir, random.choice(img_names)) for _ in range(CHUNK_SIZE)]
            upload_files = [{'image': open(img_path, 'rb')} for img_path in imgs_paths]
            start = time.time()
            futures = [executor.submit(do_request_to_endpoint, upload_file) for upload_file in upload_files]
            for future in as_completed(futures):  # нужно прожать as_completed чтобы все фьючи завершились
                future.result().json()
            finish = time.time()
            perfomance = CHUNK_SIZE / (finish - start)
            perfomances.append(perfomance)
            logging.info(f'Perfomance: {perfomance:.2f} imgs/s')
    logging.info(f'Mean perfomance: {np.mean(perfomances):.2f} img/s with std: {np.std(perfomances):.2f}')


if __name__ == '__main__':
    init_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', help='url to fastapi endpoint', default='http://0.0.0.0:5003/predict_proba')
    parser.add_argument('--threads', default=8, type=int)
    parser.add_argument('--imgs_dir', default=os.path.join(PROJECT_PATH, 'imgs'))
    parser.add_argument('--n_repeats', default=10, type=int)
    args = parser.parse_args()
    run_stress(args.imgs_dir, args.url, args.threads, args.n_repeats)
