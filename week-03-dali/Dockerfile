FROM nvcr.io/nvidia/pytorch:22.06-py3

RUN apt-get update
RUN apt-get install libgl1-mesa-glx  -y
RUN pip install --no-cache-dir ipywidgets
RUN jupyter nbextension enable --py widgetsnbextension
RUN pip install --no-cache-dir cupy-cuda112
RUN pip install --no-cache-dir pytorch-lightning albumentations timm kornia opencv-python-headless==4.4.0.44

WORKDIR /workspace/project
CMD [ "/bin/bash" ]
