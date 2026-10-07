"""Download the four public benchmarks used in FedMamba-ChebKAN from their official sources.

usage: python download_datasets.py small      # CWRU + C-MAPSS
       python download_datasets.py paderborn  # 32 bearing archives, extracted, archives removed after a verified extraction
       python download_datasets.py mimii      # fan -6/0/6 dB from Zenodo, md5-checked, extracted one at a time
Resumable: re-running skips finished files.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.request
import zipfile

F_ROOT = r"F:/Training_Data/FedMamba_ChebKAN"
D_ROOT = r"D:/Training_Data/FedMamba_ChebKAN"
UNRAR = r"C:/Program Files/WinRAR/UnRAR.exe"
LOG = os.path.join(F_ROOT, "download_log.txt")


def log(msg):
    print(msg, flush=True)
    os.makedirs(F_ROOT, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def curl(url, dest, expected_size=None):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if expected_size and os.path.exists(dest) and os.path.getsize(dest) == expected_size:
        return
    for attempt in range(5):
        r = subprocess.run(["curl", "-sSL", "--fail", "--retry", "5", "--retry-delay", "5", "-C", "-", "-o", dest, url])
        if r.returncode in (0, 33) and (not expected_size or os.path.getsize(dest) == expected_size):
            return
        log(f"  retry {attempt + 1} for {url} (rc={r.returncode})")
    raise RuntimeError(f"download failed: {url}")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def small():
    # CWRU: 12 kHz drive-end fault data + normal baseline (official case.edu pages)
    for page, sub in (("12k-drive-end-bearing-fault-data", "12k_drive_end"), ("normal-baseline-data", "normal_baseline")):
        html = urllib.request.urlopen(f"https://engineering.case.edu/bearingdatacenter/{page}").read().decode("utf-8", "ignore")
        links = sorted(set(re.findall(r'href="(https://engineering\.case\.edu/sites/default/files/[^"]+\.mat)"', html)))
        log(f"CWRU {sub}: {len(links)} files")
        for u in links:
            dest = os.path.join(F_ROOT, "CWRU", sub, u.rsplit("/", 1)[1])
            if not os.path.exists(dest) or os.path.getsize(dest) < 1000:
                curl(u, dest)
    # C-MAPSS: NASA PCoE mirror (zip containing CMAPSSData.zip)
    z = os.path.join(F_ROOT, "C-MAPSS", "turbofan_engine_degradation.zip")
    curl("https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip", z)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(os.path.join(F_ROOT, "C-MAPSS"))
    for root, _, files in os.walk(os.path.join(F_ROOT, "C-MAPSS")):
        for fn in files:
            if fn.lower() == "cmapssdata.zip":
                with zipfile.ZipFile(os.path.join(root, fn)) as zf:
                    zf.extractall(os.path.join(F_ROOT, "C-MAPSS", "CMAPSSData"))
    log("C-MAPSS done")


def paderborn():
    base = "https://groups.uni-paderborn.de/kat/BearingDataCenter/"
    html = urllib.request.urlopen(base).read().decode("utf-8", "ignore")
    names = sorted(set(re.findall(r'href="([A-Z0-9]+\.rar)"', html)))
    log(f"Paderborn: {len(names)} archives")
    for n in names:
        out = os.path.join(F_ROOT, "Paderborn", n[:-4])
        if os.path.isdir(out) and os.path.exists(os.path.join(out, ".done")):
            continue
        rar = os.path.join(F_ROOT, "Paderborn", "_archives", n)
        size = int(urllib.request.urlopen(urllib.request.Request(base + n, method="HEAD")).headers["Content-Length"])
        curl(base + n, rar, size)
        r = subprocess.run([UNRAR, "x", "-o+", "-inul", rar, os.path.join(F_ROOT, "Paderborn") + os.sep])
        if r.returncode != 0:
            raise RuntimeError(f"unrar failed for {n}")
        os.makedirs(out, exist_ok=True)
        open(os.path.join(out, ".done"), "w").close()
        os.remove(rar)
        log(f"  {n} ok ({size / 1e6:.0f} MB)")
    log("Paderborn done")


def mimii():
    meta = json.load(urllib.request.urlopen("https://zenodo.org/api/records/3384388"))
    for f in meta["files"]:
        key = f["key"]
        if not key.endswith("_fan.zip"):
            continue
        snr = key.replace("_fan.zip", "")
        out = os.path.join(D_ROOT, "MIMII", snr)
        if os.path.exists(os.path.join(out, ".done")):
            continue
        z = os.path.join(D_ROOT, "MIMII", "_archives", key)
        log(f"MIMII {key}: {f['size'] / 1e9:.2f} GB")
        curl(f["links"]["self"], z, f["size"])
        algo, digest = f["checksum"].split(":")
        if algo == "md5" and md5(z) != digest:
            os.remove(z)
            raise RuntimeError(f"md5 mismatch for {key}")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(out)
        open(os.path.join(out, ".done"), "w").close()
        os.remove(z)
        log(f"  {key} extracted, md5 ok")
    log("MIMII done")


if __name__ == "__main__":
    {"small": small, "paderborn": paderborn, "mimii": mimii}[sys.argv[1]]()
