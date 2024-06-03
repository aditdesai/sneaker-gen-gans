import streamlit as st

st.title("Generative Adversarial Networks")

model_option = st.selectbox(
    label="Select a model",
    options=["Deep Convolutional GAN", "Wasserstein GAN with GP", "Conditional GAN"],
    index=None,
    placeholder="Select a model..."
)

if model_option == "Conditional GAN":
    from test.cgan import Generator
    from test.cgan import plot_generated_images

    label_option = st.selectbox(
        label="Select a color",
        options=["Black", "Blue", "Red", "Yellow", "White"],
        index=None,
        placeholder="Select a color.."
    )

    label_mapping = {
        "Black": 0,
        "Blue": 1,
        "Red": 2,
        "Yellow": 3,
        "White": 4
    }

    container = st.container()
    cols = container.columns(5)

    if label_option:
        for i in range(10):
            img = plot_generated_images(label_mapping[label_option])
            cols[i % 5].image(img, caption=f"Generated Image {i + 1}", width=100)

elif model_option == "Wasserstein GAN with GP":
    from test.wgangp import Generator
    from test.wgangp import plot_generated_images

    container = st.container()
    cols = container.columns(5)

    for i in range(10):
        img = plot_generated_images()
        cols[i % 5].image(img, caption=f"Generated Image {i + 1}", width=100)

elif model_option == "Deep Convolutional GAN":
    from test.dcgan import generate_single_image

    container = st.container()
    cols = container.columns(5)

    for i in range(10):
        img = generate_single_image()
        cols[i % 5].image(img, caption=f"Generated Image {i + 1}", width=100)
