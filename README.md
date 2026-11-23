# Iris Studio

A polished, responsive Flask Iris flower classification app with live visual feedback, preset samples, accessible measurement controls, asynchronous server-side prediction, confidence bars (when supported), and mobile layout.

## Run it

```bash
pip install -r requirements.txt
python train_model.py  # ONLY if you do not already have your own iris.pkl
python app.py
```

Open http://127.0.0.1:5000.

**Using your existing model:** Copy your trusted `iris.pkl` into the project directory instead of running `train_model.py`. Feature order is **sepal length, sepal width, petal length, petal width** (all centimetres). The app accepts class labels 0/1/2 or species name strings. A model with `predict_proba` also displays actual prediction probabilities. Do not load untrusted pickle files.

`POST /api/predict` with JSON keys `sepal_length`, `sepal_width`, `petal_length`, `petal_width` returns a JSON classification. `POST /predict` remains compatible with the old HTML form workflow.
