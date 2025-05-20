# Generative Video Creation with RNN Weight Prediction

## Architecture Overview

(Placeholder Image: A diagram showing the VAE architecture.  Include a separate RNN that takes the input image as input and predicts weight updates for the VAE.)

The architecture consists of a Variational Autoencoder (VAE) acting as the generator and a Discriminator network.  A separate RNN, the `WeightPredictorRNN`, is trained to predict weight updates for the VAE based on the input image.

## Network Details

### VAE (Generator)

*   **Architecture:** Encoder (Conv2D, ReLu, MaxPool2D) -> Latent Space (Linear layers for mu/logvar) -> Decoder (ConvTranspose2D, ReLu, Sigmoid).  See `model_definitions.py` for specific layer details.
*   **Input:** Noise vector.
*   **Output:** Generated image (B, C, H, W).

### Discriminator

*   **Architecture:** (The layers are symmetric with the encoder.) See `model_definitions.py` for layer details.
*   **Input:** Real or generated image (B, C, H, W).
*   **Output:** Probability of the input being a real image.

### WeightPredictorRNN

*   **Architecture:** GRU with a Linear layer to predict weight updates.
*   **Parameters:**
    *   Input Size: 3 * VIDEO_HEIGHT * VIDEO_WIDTH (flattened image).
    *   Hidden Size: 128.
    *   Number of Layers: 2.
    *   Output Size: **Must match the number of parameters you want to adjust in the VAE**.
*   **Weight Update Prediction:** The RNN processes the input image and predicts a vector of weight *updates* that are applied to the VAE's layers.

## Training Data

*   Images of shape (B, C, H, W) = (16, 3, 128, 128).
*   The `ImageFolder` dataset from `torchvision` is used for loading images.

## Training Procedure

1.  The Discriminator is trained to distinguish between real and generated images using the combined Wasserstein loss with gradient penalty.
2.  The `WeightPredictorRNN` is trained to predict weight updates that improve the VAE's generator performance (i.e., generate images that can fool the Discriminator).
3.  The weight updates are applied to the VAE's weights *before* calculating the generator loss.
4.  The VAE's original parameters are also trained to generate realistic images.

## Usage

1.  Ensure you have PyTorch, torchvision, and tqdm installed.
2.  Organize your image data into a directory structure compatible with `ImageFolder`.
3.  Calculate the correct `WEIGHT_UPDATE_SIZE` based on the VAE parameters you want to adjust.
4.  Run the `training.py` script to train the model.
