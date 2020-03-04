import tensorflow as tf
import numpy as np
from PIL import Image

def image_parse(filename, parse_img=True):
    image_string = tf.io.read_file(filename)
    if parse_img:
        decoded = tf.image.decode_png(image_string, channels=3)
    else:
        decoded = tf.image.decode_jpeg(image_string, channels=1)
    return decoded

def parse_tif(filename):
    img = Image.open(filename)
    return np.array(img)
    

def img_resize(img, dims):
    return tf.image.resize_with_crop_or_pad(img, dims[0], dims[1])

def normalize(img):
    return tf.cast(img, tf.float32) / 255.0


def read_imgs_from_df(df, dims):
    images = np.array([
        row.img_path for row in df.itertuples()])
    masks = np.array([
        row.mask_path for row in df.itertuples()])
    
    image_ds = tf.data.Dataset.from_tensor_slices(images).map(
        lambda x: image_parse(x)
        ).map(
            lambda x: img_resize(x, dims)
            ).map(
                lambda x: normalize(x)
            )
    mask_ds = tf.data.Dataset.from_tensor_slices(masks).map(
        lambda x: image_parse(x, parse_img=False)
        ).map(
            lambda x: img_resize(x, dims)
            ).map(
                lambda x: normalize(x)
            )
    
    return tf.data.Dataset.zip((image_ds, mask_ds))

    