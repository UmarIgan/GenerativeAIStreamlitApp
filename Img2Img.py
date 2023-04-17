import streamlit as st
from diffusers import StableDiffusionImageVariationPipeline
from PIL import Image
import torch
import torchvision.transforms as transforms


st.set_page_config(page_title="Image Variation App")
cache_dir = "./models_variations"
# Load the model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
    "lambdalabs/sd-image-variations-diffusers", revision="v2.0"
    , cache_dir=cache_dir
)
sd_pipe = sd_pipe.to(device)

# Define the transforms for the input image
tform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
        ),
        transforms.Normalize(
            [0.48145466, 0.4578275, 0.40821073],
            [0.26862954, 0.26130258, 0.27577711],
        ),
    ]
)

# Define the Streamlit app
st.title("Image Variation App")
st.write("Upload an image and generate its variation")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the uploaded image
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption="Input Image", use_column_width=True)

    # Process the input image
    input_tensor = tform(input_image).to(device).unsqueeze(0)
    guidance_scale = st.slider(
        "Guidance scale",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="Controls the strength of guidance for the variation. Higher values give more structured results.",
    )
    with st.spinner("Generating variation..."):
        output_tensor = sd_pipe(input_tensor, guidance_scale=guidance_scale)["images"][0]
        output_image = transforms.ToPILImage()(output_tensor.detach())
        st.image(output_image, caption="Output Image", use_column_width=True)
