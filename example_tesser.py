import pytesseract
from PIL import Image

image = Image.open('result.jpg')

recognized_text = pytesseract.image_to_string(image, config='--psm 6')

print(recognized_text)
