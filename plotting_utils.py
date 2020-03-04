import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image as PImage
import numpy as np


def display_from_path(img_path):
  plt.figure(figsize=(10, 10))
  display_paths = [
      img_path,
      img_path.replace('images', 'masks').replace('.png', '_mask.png')
  ]
  titles = ['Input Image', 'Mask']

  for i in range(len(display_paths)):
    plt.subplot(1, len(display_paths), i+1)
    plt.title(titles[i])
    im = PImage.open(display_paths[i])
    plt.imshow(np.array(im))
  plt.show()
  
  
def display_array(display_list):
  plt.figure(figsize=(15, 15))
  title = ['Input Image', 'True Mask', 'Predicted Mask']
  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()