import logging

from pathlib import Path

logging.basicConfig(level=logging.DEBUG)

# Disable parso logging as it makes using ipython completion impossible
logging.getLogger("parso").setLevel(level=logging.INFO)

test_dir = (Path.cwd() / Path(__file__)).parent
