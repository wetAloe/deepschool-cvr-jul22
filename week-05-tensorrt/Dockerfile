FROM nvcr.io/nvidia/pytorch:22.06-py3

RUN apt-get update
RUN apt-get install libgl1-mesa-glx  -y
RUN pip install --no-cache-dir ipywidgets
RUN jupyter nbextension enable --py widgetsnbextension
RUN pip install --no-cache-dir pycuda==2022.1 segmentation_models_pytorch timm opencv-python-headless==4.4.0.44 onnxsim==0.4.7
RUN pip install onnxruntime

WORKDIR /workspace/project
CMD [ "/bin/bash" ]
