import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utilities.data import DATA_PREPERATION

data = DATA_PREPERATION().DOWNLOAD_DATA()
