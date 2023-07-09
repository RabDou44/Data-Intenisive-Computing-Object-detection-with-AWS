
from base64 import b64encode, b64decode
import os
#import tflite_runtime.interpreter as tflite
import platform
import datetime
import cv2
import time
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
import PIL
import io
from io import BytesIO
import requests

def draw_boxes(image, boxes, class_names=None, scores=None, max_boxes=10, min_score=0.1):
  """Overlay labeled boxes on an image with formatted scores and label names."""
  colors = list(ImageColor.colormap.values())

  try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                              25)
  except IOError:
    print("Font not found, using default font.")
    font = ImageFont.load_default()

  image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
  for i in range(min(boxes.shape[0], max_boxes)):

    if class_names is None and scores is None:
       display_str = ""
       color = "blue"
    else:
      display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                     int(100 * scores[i]))
      color = colors[hash(class_names[i]) % len(colors)]
    ymin, xmin, ymax, xmax = tuple(boxes[i])
    
    draw = ImageDraw.Draw(image_pil)
    im_width, im_height = image_pil.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
              (left, top)],
              width=4,
              fill="blue")
    
    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_height = [font.getsize(display_str)[1]]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_height)

    if top > total_display_str_height:
      text_bottom = top
    else:
      text_bottom = top + total_display_str_height
    # Reverse list and print from bottom to top.
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                    (left + text_width, text_bottom)],
                  fill=color)
    draw.text((left + margin, text_bottom - text_height - margin),
              display_str,
              fill="black",
              font=font)
    text_bottom -= text_height - 2 * margin

  return image_pil


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):
  """Adds a bounding box to an image."""
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
  draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=thickness,
            fill=color)

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = top + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                    (left + text_width, text_bottom)],
                   fill=color)
    draw.text((left + margin, text_bottom - text_height - margin),
              display_str,
              fill="black",
              font=font)
    text_bottom -= text_height - 2 * margin



##############
### Client ###
##############

class ObjectDetectionClient:
    def __init__(self, target_url):
        self.target = target_url

    def make_request(self, images:list, t):
        return requests.post(self.target, json={"images":images, "time":t})

    def encode_image(self, image):
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        return b64encode(buffer.getvalue()).decode("utf-8")

    def read_image(self, filename):
        image = Image.open(filename)
        return image
    
    def detect_images(self, list_of_image_names:list):
        t = time.time()
        encoded_images = []
        for fn in list_of_image_names:
            encoded_images.append(self.encode_image(self.read_image(fn)))
        
        response = self.make_request(encoded_images, t)
        return response
    
    def one_detection(self, image_filename):
        t = time.time()
        image = self.read_image(image_filename)
        encoded_image = self.encode_image(image)
        response = self.make_request([encoded_image], t)
        return response
    
    def show_boxes(self, image, boxes):
        if isinstance(image,str):
           image = self.read_image(image)
        if isinstance(image, PIL.JpegImagePlugin.JpegImageFile):
           image = np.asarray(image)
        if not isinstance(boxes, np.ndarray):
           boxes = np.array(boxes)
        return draw_boxes(image, boxes)