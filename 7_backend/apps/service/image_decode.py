import json
from PIL import Image
from io import BytesIO
import base64

def image_decoding(json_data):
    img = json_data['img']
    img = base64.b64decode(img)
    img = BytesIO(img)
    img = Image.open(img)
    img.save("./apps/resource/query/query.jpg", "JPEG")
    return img