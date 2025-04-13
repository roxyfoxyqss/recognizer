FROM python:3.10-slim
RUN apt update
RUN apt upgrade
RUN apt install tesseract-ocr
RUN apt install libtesseract-dev
RUN apt install git
RUN git clone