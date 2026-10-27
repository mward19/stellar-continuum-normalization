import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from show_progress_curl_script import main as download

SCRIPTS_DIR = Path("./scripts")
DOWNLOAD_DIR = Path("./data")
MAX_WORKERS = 16

all_scripts = [p.resolve() for p in SCRIPTS_DIR.iterdir()]

DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(DOWNLOAD_DIR)

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    futures = {
        pool.submit(download, file, position=i % MAX_WORKERS + 1): file
        for i, file in enumerate(all_scripts)
    }
    for future in tqdm(as_completed(futures), total=len(futures), unit="script", position=0):
        future.result()
