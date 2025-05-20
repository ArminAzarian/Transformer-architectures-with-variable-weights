import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Utility functions ---
def create_synthetic_video(image, num_frames=16):
    """Creates a synthetic video from a single image."""
    B, C, H, W = image.shape
    video = torch.zeros((B, num_frames, C, H, W), device=image.device)
    for t in range(num_frames):
        noise = torch.randn_like(image) * 0.01 * t
        video[:, t] = torch.clamp(image + noise, 0, 1)
    return video

class VAE(nn.Module):  # Same VAE as before
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.enc1 = nn.Conv2d(3, 48, kernel_size=3, padding="same") #Explicit padding with "same" will adapt the input dimension
        self.enc2 = nn.Conv2d(48, 48, kernel_size=3, padding="same")
        self.enc3 = nn.Conv2d(48, 48, kernel_size=3, padding="same")
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)#For asymmetric stride
        #Latent space parameterization layers
        self.fc1 = nn.Linear(48 * 16 * 16, 64)  #  mean
        self.fc2 = nn.Linear(48 * 16 * 16, 64)  #  log variance
        self.fc3 = nn.Linear(64, 48 * 16 * 16) # from latent space to decoder

        # Decoder
        self.dec1 = nn.ConvTranspose2d(48, 48, kernel_size=3, padding="same")
        self.dec2 = nn.ConvTranspose2d(48, 48, kernel_size=3, padding="same")
        self.dec3 = nn.ConvTranspose2d(48, 3, kernel_size=3, padding="same")


    def encoder(self, x):
        x = F.relu(self.enc1(x))
        x = self.pool(x)
        x = F.relu(self.enc2(x))
        x = self.pool(x)
        x = F.relu(self.enc3(x))
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decoder(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = torch.sigmoid(self.dec3(x)) #Sigmoid, so all the outputs are compressed from 0 to 1
        return x

    def forward(self, x):
        #Encode
        x_encoded = self.encoder(x)
        #Latent space parameterization
        mu = F.relu(self.fc1(x_encoded))
        log_var = F.relu(self.fc2(x_encoded))
        #Sampling
        z = self.reparameterize(mu, log_var)
        #Mapping to the Decoder
        x_reconstructed = F.relu(self.fc3(z))
        x_reconstructed = x_reconstructed.reshape(-1, 48, 16, 16) # From linear to image
        #Decode
        x_reconstructed = self.decoder(x_reconstructed)
        return x_reconstructed, mu, log_var #Return also mu and log_var for the loss function calculation

class WeightPredictorRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (B, T, input_size)  (We'll treat the image as a sequence)
        out, _ = self.rnn(x)
        # out: (B, T, hidden_size)
        weight_updates = self.fc(out[:, -1, :])  # Use the last timestep's output
        # weight_updates: (B, output_size)
        return weight_updates

class Discriminator(nn.Module):  # Same Discriminator as before
    def __init__(self):
        super(Discriminator, self).__init__()

        # Encoder
        self.enc1 = nn.Conv2d(3, 48, kernel_size=3, padding="same") #Explicit padding with "same" will adapt the input dimension
        self.enc2 = nn.Conv2d(48, 48, kernel_size=3, padding="same")
        self.enc3 = nn.Conv2d(48, 48, kernel_size=3, padding="same")
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)#For asymmetric stride
        #Latent space parameterization layers
        self.fc1 = nn.Linear(48 * 16 * 16, 64)  #  mean
        self.fc2 = nn.Linear(48 * 16 * 16, 64)  #  log variance
        self.fc3 = nn.Linear(64, 1) # from latent space to decoder

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = self.pool(x)
        x = F.relu(self.enc2(x))
        x = self.pool(x)
        x = F.relu(self.enc3(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x)) # From linear to image
        return x

# --- Loss Functions ---
def wasserstein_loss(real_samples, generated_samples, discriminator):
    """Calculates the Wasserstein loss."""
    real_validity = discriminator(real_samples)
    fake_validity = discriminator(generated_samples)
    wasserstein_distance = torch.mean(real_validity) - torch.mean(fake_validity)
    return wasserstein_distance

def gradient_penalty(real_samples, generated_samples, discriminator, device="cpu"):
    """Calculates the gradient penalty."""
    alpha = torch.randn((real_samples.size(0), 1, 1, 1), device=device)
    interpolated = (alpha * real_samples + ((1 - alpha) * generated_samples)).requires_grad_(True)
    interpolated_output = discriminator(interpolated)

    grad_outputs = torch.ones(interpolated_output.size(), dtype=torch.float, device=device, requires_grad=False)
    grad_interpolated = torch.autograd.grad(
        outputs=interpolated_output,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )[0]

    grad_interpolated = grad_interpolated.view(real_samples.size(0), -1)
    grad_norm = grad_interpolated.norm(2, dim=1)
    grad_penalty = torch.mean((grad_norm - 1) ** 2)
    return grad_penalty

def combined_loss(real_samples, generated_samples, discriminator, device, lambda_gp=10):
    """Combines Wasserstein loss and gradient penalty."""
    wasserstein = wasserstein_loss(real_samples, generated_samples, discriminator)
    grad_pen = gradient_penalty(real_samples, generated_samples, discriminator, device)
    return wasserstein + lambda_gp * grad_pen
