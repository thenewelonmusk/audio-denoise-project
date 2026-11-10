"""Download DNSMOS P.835 onnx models into eval/dnsmos/.

Tries the known locations in microsoft/DNS-Challenge. If the URLs have moved,
download manually from https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS
and place model_v8.onnx (+ p835 companion files) in ml/eval/dnsmos/.
"""

import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "eval" / "dnsmos"
FILES = {
    # sig_bak_ovr style single-file v8 model used by the reference dnsmos_local.py
    "model_v8.onnx": "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/model_v8.onnx",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for fname, url in FILES.items():
        dest = OUT / fname
        if dest.exists():
            print(f"exists: {dest}")
            continue
        try:
            print(f"downloading {url} ...")
            urllib.request.urlretrieve(url, dest)
            print(f"ok -> {dest}")
        except Exception as exc:
            print(f"FAILED: {exc}\nDownload manually (see module docstring).")
            dest.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
