import os
from argparse import Namespace
from os import path as osp

from model_archiver.model_packaging import package_model
from model_archiver.model_packaging_utils import ModelExportUtils
from constants import PROJECT_PATH

DIR_PATH = os.path.dirname(__file__)


def generate_mar(torch_checkpoint_path: str, mar_checkpoint_dir: str, model_name: str):
    os.makedirs(mar_checkpoint_dir, exist_ok=True)
    handler_path = osp.join(DIR_PATH, 'model_handler.py')
    args = Namespace(**{
        'model_file': None,
        'model_name': model_name,
        'version': '1.0',
        'serialized_file': torch_checkpoint_path,
        'handler': handler_path,
        'export_path': mar_checkpoint_dir,
        'force': False,
        'extra_files': None,
        'requirements_file': osp.join(DIR_PATH, 'requirements.txt'),
        'runtime': 'python',
        'archive_format': 'default'
    })
    manifest = ModelExportUtils.generate_manifest_json(args)
    package_model(args, manifest)


if __name__ == '__main__':
    generate_mar(
        os.path.join(PROJECT_PATH, 'weights', 'genre_classifier.pt'),
        os.path.join(PROJECT_PATH, 'model_mars'),
        'my_model',
    )
