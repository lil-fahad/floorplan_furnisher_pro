
# Floorplan Furnisher Pro

تطبيق احترافي يأخذ **مخطط منزل** (صورة)، يحلّله دلاليًا، ويبني **توزيع أثاث ذكي**.

## Quickstart
```bash
pip install -r requirements.txt
streamlit run apps/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
uvicorn furniture_ai.api.server:app --host 0.0.0.0 --port 8000 --reload
```

### Alternative ways to launch the Streamlit UI

**Docker**

```bash
docker build -f Dockerfile.streamlit -t floorplan-furnisher-ui .
docker run -p 8501:8501 floorplan-furnisher-ui
```

**Makefile**

```bash
make -f Makefile.make ui
```

> Adjust the segmenter model path in the sidebar or provide model weights so
> `run_furnish` can locate `models/segmenter/best.pt`.

## Kaggle credentials and datasets

Downloadable training data is hosted on Kaggle. Place your `kaggle.json`
in the project root or `~/.kaggle/`, or export `KAGGLE_USERNAME` and
`KAGGLE_KEY` before running scripts.

`scripts/train_detector.py` can fetch a dataset automatically when you
pass its slug:

```bash
python scripts/train_detector.py --dataset-slug user/dataset
```

If the target directory is empty the dataset is downloaded into
`--data-root` and unzipped.
