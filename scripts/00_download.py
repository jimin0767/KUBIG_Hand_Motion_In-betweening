"""SignSparK How2Sign LMDB 다운로드 (HuggingFace).

CSL-Daily는 0816 회의 결정(저자 공지 미해결 버그)으로 받지 않는다.
받는 것: train/dev/test How2Sign_reopt_*.lmdb  = 약 8.9GB
"""
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

REPO = "LionelLow/SignSparK_data"
DEST = Path(r"C:\tmp\signspark_data")

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    p = snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        local_dir=str(DEST),
        allow_patterns=["*How2Sign*", "README.md"],
        max_workers=4,
    )
    print("DONE ->", p)
    for f in sorted(DEST.rglob("*.mdb")):
        print(f"{f.stat().st_size/1e9:6.2f} GB  {f}")

if __name__ == "__main__":
    sys.exit(main())
