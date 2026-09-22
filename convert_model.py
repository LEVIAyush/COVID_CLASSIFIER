"""
One-off conversion of your already-trained model. No retraining, no dataset.

  python convert_model.py "C:/Users/levia/OneDrive/Desktop/ML/Covid.h5"

Run it in the SAME Python environment you trained in (same TensorFlow version),
so the old .h5 loads exactly as it did before.

Outputs, in ./models:
  Covid_hybrid_v2.keras   the current Keras format (recommended)
  Covid_hybrid_v2.tflite  optional, ~half the size (float16), via --tflite
"""
import argparse
import os

import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("source", help="path to the trained .h5 file")
parser.add_argument("--tflite", action="store_true", help="also write a float16 TFLite file")
args = parser.parse_args()

os.makedirs("models", exist_ok=True)
model = tf.keras.models.load_model(args.source, compile=False)
print("Input shape:", model.input_shape, "| Output shape:", model.output_shape)

model.save("models/Covid_hybrid_v2.keras")
print("Wrote models/Covid_hybrid_v2.keras")

probe = np.random.rand(1, 224, 224, 3).astype("float32")
reference = float(model.predict(probe, verbose=0)[0][0])

if args.tflite:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    with open("models/Covid_hybrid_v2.tflite", "wb") as f:
        f.write(converter.convert())

    interp = tf.lite.Interpreter(model_path="models/Covid_hybrid_v2.tflite")
    interp.allocate_tensors()
    interp.set_tensor(interp.get_input_details()[0]["index"], probe)
    interp.invoke()
    lite = float(interp.get_tensor(interp.get_output_details()[0]["index"])[0][0])
    print(f"Keras output {reference:.4f} vs TFLite output {lite:.4f}")
    print("Wrote models/Covid_hybrid_v2.tflite")

for name in sorted(os.listdir("models")):
    print(f"{name}: {os.path.getsize(os.path.join('models', name)) / 1e6:.1f} MB")
