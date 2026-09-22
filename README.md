# Chest X-ray COVID screening (demo)

Student project. Not a medical device.

## Run locally
1. `python convert_model.py "path/to/Covid.h5" --tflite`  (once, in the environment you trained in)
2. `pip install -r requirements.txt`
3. `python flask_app.py`, then open http://localhost:5000

## Settings (environment variables)
| Variable | Meaning | Default |
|---|---|---|
| MODEL_PATH | model file (.keras, .h5 or .tflite) | models/Covid_hybrid_v2.keras |
| MODEL_URL | download the model here on first start if the file is missing | none |
| COVID_CLASS_INDEX | which sigmoid output means COVID (0 or 1) | 0 |

## Check COVID_CLASS_INDEX before you trust the labels
In the training script add `print(train_gen.class_indices)`. Classes are numbered
alphabetically. If it prints `{'Covid': 0, 'Normal': 1}` keep the default 0.
If COVID is 1, set `COVID_CLASS_INDEX=1`.

## Deploy
`docker build -t covid-xray . && docker run -p 7860:7860 covid-xray`
Or skip copying the model into the image: keep `models/` empty and set MODEL_URL.
