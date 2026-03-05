from model_handlers.generative_model import ImageGenerator


# --------------------------
# Training
# --------------------------
LEARNING_RATE_GENERATOR = 0.0003
LEARNING_RATE_DISCRIMINATOR = 0.0001

EPOCHES = 20
image_generator = ImageGenerator()
image_generator.fit(EPOCHES, LEARNING_RATE_GENERATOR, LEARNING_RATE_DISCRIMINATOR)