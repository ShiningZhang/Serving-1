import sys
from paddle_serving_app.reader import Sequential, URL2Image, Resize, CenterCrop, RGB2BGR, Transpose, Div, Normalize, Base64ToImage
import numpy as np
import base64, cv2

def preprocess(data):
  # data = np.fromstring(data, np.uint8)
  # im = cv2.imdecode(data, cv2.IMREAD_COLOR)
  print(data)
  print(data.shape)
  print(type(data))
  seq = Sequential([
            Resize(256), CenterCrop(224), RGB2BGR(), Transpose((2, 0, 1)),
            Div(255), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
                                True)
        ])
  img = seq(data)
  print(img.shape)
  print(type(img))
  return img