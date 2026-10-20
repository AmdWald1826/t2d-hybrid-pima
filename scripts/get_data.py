"""Download the public PIMA Indians Diabetes file and verify its SHA-256."""
import hashlib
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
SHA256 = "6bfe5d0f379d17a0e0819b996407e3c09bf80febd4287f2ed212190dfff154af"
OUT = Path(__file__).resolve().parents[1] / "data" / "pima-indians-diabetes.csv"

if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    if not OUT.exists():
        urllib.request.urlretrieve(URL, OUT)
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"Checksum mismatch: {digest}")
    print(f"OK: {OUT} (768 records, SHA-256 verified)")
