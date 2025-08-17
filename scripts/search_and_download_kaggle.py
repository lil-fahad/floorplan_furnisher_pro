
import json, subprocess, argparse
from pathlib import Path
from furniture_ai.utils.kaggle_io import ensure_kaggle_creds

KEYWORDS = ["furniture","floorplan","layout","interior","architecture","engineering","design"]

def search_kaggle(keywords):
    results = {}
    for kw in keywords:
        cmd = ["kaggle","datasets","list","-s",kw,"--csv"]
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode!=0:
            print(out.stderr); continue
        lines = out.stdout.strip().splitlines()
        for line in lines[1:]:
            cols = line.split(",")
            if cols:
                ref = cols[0]
                title = cols[1] if len(cols)>1 else ""
                results[ref] = title
    return results

def download_dataset(slug, dest="data/raw"):
    Path(dest).mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["kaggle","datasets","download","-d",slug,"-p",dest,"--unzip"])

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--download", nargs="+")
    ap.add_argument("--dest", default="data/raw")
    args = ap.parse_args()

    if not ensure_kaggle_creds():
        raise SystemExit("kaggle.json غير موجود. ضع الملف في ~/.kaggle/ أو جذر المشروع")
    if args.list:
        ds = search_kaggle(KEYWORDS)
        print(json.dumps(ds, indent=2, ensure_ascii=False))
    if args.download:
        for slug in args.download:
            print(f"Downloading {slug} ...")
            download_dataset(slug, dest=args.dest)
