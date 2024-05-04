import numpy as np
from tensorflow.keras.models import load_model


def generate_single_image():
    generator_model_path = "DCGAN/model_200.h5"
    gen_model = load_model(generator_model_path)

    noise = np.random.randn(1, 100)

    generated_image = gen_model.predict(noise)

    generated_image = 0.5 * generated_image + 0.5
    generated_image = (generated_image * 255).astype(np.uint8)

    return generated_image
