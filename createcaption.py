import re
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
import preprocessing
import load
from tensorflow.keras.preprocessing.sequence import pad_sequences



max_length =37


def create_caption_cnn_lstm(img_id, tokenizer, model, max_len_caption = max_length):
  img_f = load.vgg_model.predict(preprocessing.preprocess_image(img_id))
  start_token = load.tokenizer.word_index['<start>']
  end_token = load.tokenizer.word_index['<end>']
  tokens = []
  tokens.append(start_token)
  for i in range(max_len_caption):
    padded_tokens = pad_sequences([tokens], maxlen = max_length, padding = "post")[0]
    predicted_tokens = load.model.predict([img_f,np.array([padded_tokens])])
    highest_prob_token_index = np.argmax(predicted_tokens[0])
    tokens.append(highest_prob_token_index)
    #print(tokens)
    if highest_prob_token_index == end_token:
      break

  generated_caption = ""
  for token in tokens[1:-1]:
    generated_caption += " " + load.tokenizer.index_word[token]
  
  return "caption: "+ generated_caption.strip().capitalize()

def create_caption_llm(img_id, model, processor):
  path = 'images/'+img_id
  print(path)
  raw_image = Image.open(path).convert('RGB')
  inputs = processor(raw_image, text = "Caption:", return_tensors="pt")
  out = model.generate(**inputs, max_new_tokens = max_length)
  caption_llm = processor.decode(out[0], skip_special_tokens=True)
  return caption_llm.capitalize()

def streamlit_app():
    # # logo
    # image = Image.open('logo_path')
    # st.image(image, width=200)
    # title
    st.title('Image Captioning')
    # user input
    img_list = ['1.jpg','2.jpg','3.jpg','4.jpg','5.jpg', '6.jpg', '7.jpg', '8.jpg', '9.jpg', '10.jpg']
    img_selected = str(st.selectbox('Select an Image', img_list))

    
    # sidebar: choose model
    st.sidebar.title('Model')
    model_list = ["VGG+LSTM", "LLM"]
    model_selected = st.sidebar.selectbox('Model', model_list)

    # #select type of method of the selected task
    # if model_selected == "Inception+LSTM":
    #     parameter = st.sidebar.selectbox("Keyword extraction", ["Word cloud", "Spacy", "Yake"])
        
    return img_selected, model_selected


def output_caption():
    img_selected, model_selected = streamlit_app()
    image = Image.open('images/'+ img_selected)
    st.image(image, width=400)

    if model_selected =="VGG+LSTM":
       caption = create_caption_cnn_lstm(img_id = img_selected, tokenizer= load.tokenizer, model = load.model, max_len_caption = max_length)
       st.write(caption)
    
    else:
       caption_llm = create_caption_llm(img_id = img_selected, model = load.model_llm, processor= load.processor)
       st.write(caption_llm)
       