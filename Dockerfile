FROM python:3.10-slim
RUN apt update
RUN apt upgrade
RUN apt install tesseract-ocr
RUN apt install libtesseract-dev
RUN apt install git
RUN git clone https://github.com/roxyfoxyqss/recognizer.git -b main
RUN pip install -r requirements.txt
RUN pip install Flask
RUN pip install pytesseract
ENTRYPOINT [ "python3", "number_recognize.py"]