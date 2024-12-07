# 🚀 Installing PyTorch and TorchVision on JetPack 6.1  
#### *For NVIDIA Jetson Orin AGX Developer Kit (64GB)*  

---

## 🔍 Overview

Setting up PyTorch and TorchVision on JetPack 6.1 can be a challenging task due to limited and often outdated documentation. This guide will walk you through installing PyTorch and building TorchVision from source to get your NVIDIA Jetson Orin AGX Developer Kit (64GB) ready for AI/ML tasks.

---

## 🤔 Why This Guide?

I faced several challenges while setting up my device for YOLO-based real-time object detection. The lack of accurate guidelines from official sources, even on NVIDIA forums, made it a time-consuming process. Here's a streamlined guide based on my experience to save you the hassle.

---

## 🛠 Steps to Install PyTorch and TorchVision

### 1️⃣ Prerequisites
Update your system and install required dependencies:
```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y python3-pip libjpeg-dev libpng-dev libtiff-dev
```
Set up a Python virtual environment (recommended):

python3 -m venv ~/env
source ~/env/bin/activate

### 2️⃣ Install PyTorch

Download the compatible PyTorch wheel from NVIDIA:

    Visit NVIDIA PyTorch Downloads.
    Select the PyTorch wheel for JetPack 6.1:
        Example file: torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl

Install PyTorch:

pip install /path/to/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl

### 3️⃣ Build TorchVision from Source

Since NVIDIA doesn’t provide a prebuilt TorchVision wheel for JetPack 6.1, you’ll need to build it from source.
Clone the Repository:

git clone https://github.com/pytorch/vision.git
cd vision

Checkout the Compatible Version:

Make sure the TorchVision version aligns with your PyTorch version:

git checkout tags/v0.20.0

Build and Install TorchVision:

python3 setup.py install

    ⚠️ Note: Ensure you’ve installed dependencies (libjpeg-dev, libpng-dev, libtiff-dev) before building.

### 4️⃣ Verify Installation

Run this script to confirm the setup:

import torch
import torchvision

print("Torch Version:", torch.__version__)
print("TorchVision Version:", torchvision.__version__)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA Device:", torch.cuda.get_device_name(0))

✔️ Expected Output:

    Torch Version: 2.5.0a0+872d972e41.nv24.08
    TorchVision Version: 0.20.0
    CUDA Available: True
    CUDA Device: Orin

## ⚙️ Why Build TorchVision from Source?

Building TorchVision from source ensures:

    Compatibility with your specific PyTorch version.
    Flexibility to customize or optimize for your needs.

## 🐛 Troubleshooting
Common Issues:

    Version Mismatch:
        Verify TorchVision matches your installed PyTorch version.
        Use git checkout to get the correct version from GitHub.

    Missing Dependencies:
        Ensure dependencies like libjpeg-dev, libpng-dev, and libtiff-dev are installed:

    sudo apt-get install libjpeg-dev libpng-dev libtiff-dev

Build Errors:

    Update setuptools and pip:

        pip install --upgrade pip setuptools

## 🎉 Conclusion

Follow this guide to efficiently set up PyTorch and TorchVision on JetPack 6.1, leveraging the full potential of NVIDIA Jetson Orin AGX for AI/ML projects. Save and share this guide with others to streamline their setup process. Happy coding! 🚀

-------------------
💡 Author

👨‍💻 Azimjon Akhtamov
AI Research Assistant | Graduate Computer Science Student at CBNU | South Korea



⭐ Contribute

Fork this repository and submit pull requests for improvements or additional issues. Let’s help the community together! 💪
