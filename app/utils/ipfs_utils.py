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
    
gateways = ["https://ipfs.io/", "https://chocolate-tremendous-possum-944.mypinata.cloud/"]

async def download_weights_from_ipfs_async(session, ipfs_cid: str, client_id: int) -> str:
    """
    Download the file from IPFS and return the temp file path.
    """
    # If already has file, return the file path, for testing purpose
    for file in TEMP_FILE_PATH.iterdir():
        if ipfs_cid in file.name:
            print("File exists: ", file)
            return client_id, str(file)
    

    file = None
    for gateway in gateways:
        try:
            add_url = f"{gateway}ipfs/{ipfs_cid}"
            async with session.get(add_url) as r:
                if r.status_code == 200:
                    file = await r.read()
                    file_type = r.headers['Content-Type']
                    print(add_url + "Used")
                    logging.info(f"Successfully fetched the file from {gateway}")
                    break  # Successfully fetched the file, break out of the loop
                else: continue
        except:
            continue  # Try the next gateway

    if file is None:
        raise HTTPException(description="Failed to download the file from IPFS")

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

def download_file_from_ipfs(ipfs_cid: str, client_id) -> str:
    """
    Download the file from IPFS and return the temp file path.
    """
    # If already has file, return the file path, for testing purpose
    for file in TEMP_FILE_PATH.iterdir():
        if ipfs_cid in file.name:
            print("File exists: ", file)
            return client_id, str(file)
    
    # Try all address unless all disconnected then throw errors
    for gateway in gateways:
        try:
            url = f"{gateway}ipfs/{ipfs_cid}"
            r = requests.get(url)
            if r.status_code == 200:
                print(url + "Used")
                logging.info(f"Successfully fetched the file from {gateway}")
                break  # Successfully fetched the file, break out of the loop
        except:
            continue  # Try the next gateway
    else:  # No break encountered
        raise HTTPException(description="Failed to download the file from IPFS")

    # Get the file type and save the file
    file_type = r.headers['Content-Type']
    print(file_type)
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
    return client_id, str(TEMP_FILE_PATH) + "/" + filename
    
def download_model_from_ipfs(ipfs_cid: str):
    """
    Download the model from IPFS and return the model object.
    """
    model = download_file_from_ipfs(ipfs_cid, 0)[1]
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

def upload_to_ipfs(filepath):
    """
    Upload the file to IPFS and return the CID.
    """
    url = "https://api.pinata.cloud/pinning/pinFileToIPFS"
    headers = {"Authorization": "Bearer " + os.getenv("PINATA_API_KEY")}
    files = {
        "file": open(filepath, 'rb')
    }
    r = requests.post(url, headers=headers, files=files)
    if r.status_code == 200:
        return r.json()['IpfsHash']
    else:
        raise HTTPException(description="Failed to upload the file to IPFS")

# Testing the function
if __name__ == "__main__":
    # res = download_file_from_ipfs("QmdfzatZMtMmaWMTNsJuvCQcBbHAQFSGTrYLi6CiZ5fWTi", 0)
    # print(res)
    temp = "temp_weights.h5"
    filepath = TEMP_FILE_PATH / temp
    cid = upload_to_ipfs(filepath)
    print(cid)



    
