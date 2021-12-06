from pathlib import Path
import json
from collections import OrderedDict

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)
