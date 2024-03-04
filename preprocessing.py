import re
import tensorflow as tf
import numpy as np
import load

vgg_size = (224,224)
#inception_size = (299,299)

def preprocess1_caption(x):
  x = x.lower()
  x = x.split(".")[0]
  x = re.sub("[\"]","",x)
  x = x.strip()
  x = "<start> " +x +" <end>"
  return x

def preprocess2_caption(capts):
  caption_list = [preprocess1_caption(c) for c in capts]
  caption_seq = load.tokenizer.texts_to_sequences(caption_list)
  return caption_seq

def preprocess_image(img_id):
    img_path = "images/"+img_id
    img = tf.io.read_file(img_path)  # Read image file
    img = tf.image.decode_jpeg(img, channels=3)  # Decode JPEG-encoded image
    img = tf.image.resize(img, vgg_size)  # Resize image
    img = np.expand_dims(img, axis=0)
    return img