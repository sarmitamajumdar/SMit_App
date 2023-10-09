#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
# ### App File

# %%
!pip install tensorflow

 import gradio as gr
import cv2
import tensorflow as tf

def predict_input_image(img):

    '''To predict Diabetic Retinopathy images'''
    
    model = tf.keras.models.load_model(r"/Users/Sarmita Majumdar/Desktop/SMit_App/model00000002-0.9092040061950684.h5") 
    Retina_classes:str = ['DR', 'No_DR']
    img_resize = img.reshape(-1,224,224,3)
    prediction:float=model.predict(img_resize)[0]
    return {Retina_classes[i]: float(prediction[i]) for i in range(2)}

def  GUI():

    '''1. Gradio - Graphical user interface development tool and easy to access anyone

    2. Instead of using HTML/ CSS/ JS -> I am using Gradio'''
   
    # Input shape represent by Gradio
    image = gr.inputs.Image(shape=(224,224))
    
    # Number of Classes using by Gradio
    label = gr.outputs.Label(num_top_classes=2)
    
    # All are in one place to serve Browser!
    gr.Interface(fn=predict_input_image, inputs=image, outputs=label,interpretation='default').launch(debug='True', share=True)

GUI()


# %%





# %%
