"""Portable environment bridge for the preserved benchmark adapters."""
import json
import os
from pathlib import Path
import urllib.request

ROOT = Path(os.environ['QBENCH_WORK_DIR'])


def paths(gpu=None, expert=False):
    return os.environ['QBENCH_MODEL_PATH'], os.environ['QBENCH_TOKENIZER_PATH']


def server_context_capacity(gpu=None):
    with urllib.request.urlopen(os.environ['QBENCH_ENDPOINT'] + '/v1/models', timeout=15) as response:
        return int(json.load(response)['data'][0]['max_model_len'])
