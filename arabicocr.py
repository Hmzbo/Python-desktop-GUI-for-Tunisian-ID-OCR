from easyocr import Reader
import string
def cleanup_text(text):
	# strip out non-ASCII text so we can draw the text on the image
	# using OpenCV
  result = text.replace('\n\x0c','')
  result = result.replace('\u200f','')
  result = result.replace('\u200e','')
  result = result.translate(str.maketrans('','', string.punctuation))
  result = result.translate(str.maketrans('','', ''.join(list(string.ascii_lowercase))))
  result = result.translate(str.maketrans('','', ''.join(list(string.ascii_uppercase)))).strip()
  return result

def arabic_ocr(image):
  # break the input languages into a comma separated list
  langs = "ar,en".split(",")
  #print("[INFO] OCR'ing with the following languages: {}".format(langs))
# OCR the input image using EasyOCR
  print("[INFO] OCR'ing input image...")
  reader = Reader(langs)
  results = reader.readtext(image)
  print("Results:")
  print(results)
  return results
