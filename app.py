from model_handlers.generative_model import ImageGenerator


# --------------------------
# Training
# --------------------------
LEARNING_RATE = 0.0001
EPOCHES = 20

image_generator = ImageGenerator()
image_generator.fit(EPOCHES, LEARNING_RATE)