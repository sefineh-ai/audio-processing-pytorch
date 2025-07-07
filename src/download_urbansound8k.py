import os
import zipfile
import requests
from tqdm import tqdm

URBANSOUND8K_URL = "https://zenodo.org/record/1203745/files/UrbanSound8K.zip?download=1"
OUT_ZIP = "UrbanSound8K.zip"
OUT_DIR = "UrbanSound8K"


def download_file(url, out_path):
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(out_path, 'wb') as file, tqdm(
        desc=out_path,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

if __name__ == "__main__":
    if not os.path.exists(OUT_ZIP):
        print(f"Downloading UrbanSound8K dataset...")
        download_file(URBANSOUND8K_URL, OUT_ZIP)
    else:
        print(f"{OUT_ZIP} already exists.")
    if not os.path.exists(OUT_DIR):
        print(f"Extracting {OUT_ZIP}...")
        extract_zip(OUT_ZIP, ".")
    else:
        print(f"{OUT_DIR} already extracted.")
    print("Done.") 