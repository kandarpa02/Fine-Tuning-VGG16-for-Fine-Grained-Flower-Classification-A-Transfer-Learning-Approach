import os
import gdown

WEIGHTS_URL = "https://drive.google.com/file/d/1ExRLjPnPcV5-XSvQKlv_izYRfHtlAVUB/view?usp=drive_link"
WEIGHTS_PATH = "Fine-Tuning-VGG16-for-Fine-Grained-Flower-Classification-A-Transfer-Learning-Approach/classifier/model/model.pth"

if not os.path.exists(WEIGHTS_PATH):
    gdown.download(WEIGHTS_URL, WEIGHTS_PATH, quiet=False, fuzzy = True)
