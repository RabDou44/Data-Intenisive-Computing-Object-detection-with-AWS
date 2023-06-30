
from base64 import b64encode, b64decode
from PIL import Image
from PIL import ImageDraw
import os
import detect
import tflite_runtime.interpreter as tflite
import platform
import datetime
import cv2
import time
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
import io
from io import BytesIO
from flask import Flask, request, Response, jsonify
import requests

def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
  """Overlay labeled boxes on an image with formatted scores and label names."""
  colors = list(ImageColor.colormap.values())

  try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                              25)
  except IOError:
    print("Font not found, using default font.")
    font = ImageFont.load_default()

  for i in range(min(boxes.shape[0], max_boxes)):
    if scores[i] >= min_score:
      ymin, xmin, ymax, xmax = tuple(boxes[i])
      display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                     int(100 * scores[i]))
      color = colors[hash(class_names[i]) % len(colors)]
      image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
      draw_bounding_box_on_image(
          image_pil,
          ymin,
          xmin,
          ymax,
          xmax,
          color,
          font,
          display_str_list=[display_str])
      np.copyto(image, np.array(image_pil))
  return image


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

    def make_request(self, images:list[BytesIO]):
        requests.post(self.target, data={"data":images})

    def encode_image(self, image):
        return io.BytesIO(b64encode(image))

    def read_image(self, filename):
        image = cv2.imread(filename)
        return image
    
    def detect_images(self, list_of_image_names:list[str]):
        encoded_images = []
        for fn in list_of_image_names:
            encoded_images.append(self.encode_image(self.read_image(fn)))
        
        response = self.make_request(encoded_images)
        return response
    
    def one_detection(self, image_filename):
        image = self.read_image(image_filename)
        encoded_image = self.encode_image(image)
        response = self.make_request([encoded_image])
        return response
    
    def show_boxes(self, image, bounding_boxes):
        if isinstance(image,str):
           image = self.read_image(image)
        scores = bounding_boxes["scores"]
        labels = bounding_boxes["labels"]
        boxes = bounding_boxes["boxes"]
        return draw_boxes(image, boxes, labels, scores)