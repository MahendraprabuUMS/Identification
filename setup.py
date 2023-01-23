import setuptools
import os

os.system("pip install git+https://github.com/openai/CLIP.git") 
os.system("pip install sentence-transformers")


setuptools.setup(
    name="Identification",
    version="1.0",
    author="MP",
    author_email="",
    description="Zeroshot Image Identification",
    long_description="Zero Shot Image Identification equivalent for HuggingFace Zero Shot Text Classification - By Prithivi Da",
    url="https://github.com/PrithivirajDamodaran/ZSI.git",
    packages=setuptools.find_packages(),
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: MIT",
        "Operating System :: OS Independent",
    ],
)
