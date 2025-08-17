
import os, json, stat
from pathlib import Path
def ensure_kaggle_creds():
    default = Path.home() / ".kaggle" / "kaggle.json"
    local = Path("kaggle.json")
    if local.exists():
        default.parent.mkdir(parents=True, exist_ok=True)
        default.write_bytes(local.read_bytes())
        os.chmod(default, stat.S_IRUSR | stat.S_IWUSR)
        return True
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        data = {"username": os.environ["KAGGLE_USERNAME"], "key": os.environ["KAGGLE_KEY"]}
        default.parent.mkdir(parents=True, exist_ok=True)
        default.write_text(json.dumps(data))
        os.chmod(default, stat.S_IRUSR | stat.S_IWUSR)
        return True
    return default.exists()
