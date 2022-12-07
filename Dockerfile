FROM python:3.10

RUN pip install torch torchvision
RUN pip install opencv-contrib-python
RUN pip install scikit-learn
RUN pip install jupyterlab
RUN pip install matplotlib
RUN pip install imutils
RUN pip install pillow==6.2.1

RUN apt-get update -y && apt-get install libgl1 -y