import logging
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

########## Configurations ##########
_ = load_dotenv(find_dotenv())
LOG_PATH = Path(__file__).parent.parent / os.getenv("LOG_FOLDER")
if not LOG_PATH.exists():
    LOG_PATH.mkdir()
####################################

def setup_logging(log_file_name):
    logger = logging.getLogger()
    # logging.basicConfig(level=logging.DEBUG,
    #                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    #                     datefmt='%Y-%m-%d %H:%M:%S')
    
    file_handler = logging.FileHandler(LOG_PATH / log_file_name)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(stream_handler)