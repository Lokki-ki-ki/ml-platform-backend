import logging
import asyncio
import requests
import tensorflow as tf
import hashlib
from tensorflow.keras import models
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from werkzeug.exceptions import HTTPException
import os

########## Configurations ##########
_ = load_dotenv(find_dotenv())
FOLDER = os.getenv("TEMP_FOLDER")
TEMP_FILE_PATH = Path(__file__).parent.parent / FOLDER
if not TEMP_FILE_PATH.exists():
    TEMP_FILE_PATH.mkdir()
####################################

async def download_weights_from_ipfs_async(session, ipfs_cid: str, client_id: int) -> str:
    """
    Download the file from IPFS and return the temp file path.
    """
    add_url = "https://ipfs.io/ipfs/" + ipfs_cid
    async with session.get(add_url) as r:
        if r.status != 200:
            raise HTTPException(description="Failed to download the file from IPFS", code=r.status)
        file = await r.read()
        file_type = r.headers['Content-Type']
        if file_type == "application/json":
            filename = ipfs_cid + ".json"
        elif file_type == "text/plain":
            filename = ipfs_cid + ".txt"
        elif file_type == "application/octet-stream":
            filename = ipfs_cid + ".h5"
        else:
            raise HTTPException("Unsupported file type", 501)
        # TODO: To delete the file after the session is closed
        with open(TEMP_FILE_PATH / filename, 'wb') as f:
            f.write(file)
            f.close()
        return client_id, str(TEMP_FILE_PATH) + "/" + filename

def download_file_from_ipfs(ipfs_cid: str) -> str:
    """
    Download the file from IPFS and return the temp file path.
    """
    add = "https://ipfs.io/ipfs/" + ipfs_cid
    # logging.info("Downloading the file from IPFS:%s".format(add))
    r = requests.get(add)
    if r.status_code != 200:
        # logging.error("Failed to download the file. Status code: ", r.status_code)
        raise HTTPException(description="Failed to download the file from IPFS", code=r.status_code)
    # Get the file type and save the file
    file_type = r.headers['Content-Type']
    # logging.info("File type:%s".format(file_type))
    if file_type == "application/json":
        filename = ipfs_cid + ".json"
    elif file_type == "text/plain":
        filename = ipfs_cid + ".txt"
    elif file_type == "application/octet-stream":
        filename = ipfs_cid + ".h5"
    else:
        raise HTTPException("Unsupported file type", 501)
    with open(TEMP_FILE_PATH / filename, 'wb') as f:
        f.write(r.content)
        f.close()
    return str(TEMP_FILE_PATH) + "/" + filename
    
def download_model_from_ipfs(ipfs_cid: str):
    """
    Download the model from IPFS and return the model object.
    """
    model = download_file_from_ipfs(ipfs_cid)
    if model:
        # model = models.model_from_json(model)
        model = models.load_model(model)
    return model

def hash_file(filepath, block_size=65536):
    """
    Hash a file using SHA-256 and return the hexadecimal hash value.
    """
    hash_sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            hash_sha256.update(block)
        f.close()
    return hash_sha256.hexdigest()

# Testing the function
if __name__ == "__main__":
    res = download_file_from_ipfs("QmT5JHVS6SQDvnqucZjqNHsQbJWEJmv1vadxzKMdDwL8Ah")
    print(res)


    
