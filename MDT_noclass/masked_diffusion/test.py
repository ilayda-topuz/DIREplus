import os

if  os.environ.get("DIFFUSION_TRAINING_TEST", ""):
    print("True", os.environ.get("DIFFUSION_TRAINING_TEST", ""))
else:
    print("False", os.environ.get("DIFFUSION_TRAINING_TEST", ""))