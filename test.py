from ultralytics import YOLO

# Load model
model = YOLO("best.pt")

# Path of input image
source_path = r"C:\Users\asus\OneDrive\Documents\testing\test2.jpg"

# Run prediction
results = model.predict(
    source=source_path,
    conf=0.25,
    save=True
)

print("✅ Prediction completed! Check the 'runs/detect' folder for output.")
