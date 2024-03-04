import tensorflow as tf
import pickle
from transformers import BlipProcessor, BlipForConditionalGeneration, GenerationConfig

artifact_path = 'tools/'

model = tf.keras.models.load_model(artifact_path +"model.h5")
#model.compile(loss='categorical_crossentropy', optimizer='adam')


vgg_model = tf.keras.models.load_model(artifact_path +"vgg_model.h5")

with open(artifact_path +'tokenizer.pkl', 'rb') as f:
   tokenizer = pickle.load(f)


processor = BlipProcessor.from_pretrained(artifact_path +"processor")
model_llm = BlipForConditionalGeneration.from_pretrained(artifact_path +"model_llm")
