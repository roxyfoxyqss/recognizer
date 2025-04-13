FROM python:3.10-slim
RUN apt update
RUN apt upgrade
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
RUN apt-get update && apt-get install libgl1 -y
RUN apt install tesseract-ocr -y
RUN apt install libtesseract-dev -y
RUN apt install git -y
WORKDIR /yolov5/
COPY . .
RUN pip install -r requirements.txt
RUN pip install Flask
RUN pip install transformers==4.30.0 numpy==1.24.0
ENTRYPOINT [ "python3", "number_recognize.py"]