from setuptools import setup

setup(
    name="masked-diffusion",
    py_modules=["masked_diffusion"],
    install_requires=["blobfile>=1.0.5", "torch==1.13.1", "torchvision==0.14.1", "torchaudio==0.13.1", "tqdm"],
)
