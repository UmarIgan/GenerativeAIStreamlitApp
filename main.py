import streamlit as st
from PIL import Image
from io import BytesIO
from diffusers import DiffusionPipeline
from transformers.utils import FLAX_WEIGHTS_NAME
FLAX_WEIGHTS_NAME = 'flax_model.msgpack'

st.set_page_config(page_title="Text to Image App")

# Download the model once and save it to a local directory
model_id = "CompVis/ldm-text2im-large-256"
ldm = DiffusionPipeline.from_pretrained(model_id, cache_dir="./models")

st.title("Image Generation App")

# Get user input
prompt = st.text_input("Enter a prompt for image generation", "A painting of a squirrel eating a burger")

if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        # Run the model to generate images
        images = ldm([prompt], num_inference_steps=50, eta=0.3, guidance_scale=6).images

        # Display the images
        for idx, image in enumerate(images):
            st.image(image, caption=f"Generated Image {idx+1}")
