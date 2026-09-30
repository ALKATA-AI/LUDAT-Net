# In the name of Allah, the most compassionate, the most merciful

import torch
import whisper

print(f"Is CUDA available? {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")

model = whisper.load_model("base", device="cuda") # base or large-v3
print("Model loaded successfully on GPU!")
