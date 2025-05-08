# clustering_mnist_autoencoder.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import numpy as np

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Dataset Loading and Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
data_loader = DataLoader(mnist_data, batch_size=128, shuffle=True)

# 2. Define Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=64):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# 3. Initialize Model, Loss, Optimizer
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 4. Training Loop
epochs = 10
print("Training the autoencoder...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, _ in data_loader:
        images = images.view(images.size(0), -1).to(device)
        encoded, decoded = model(images)
        loss = criterion(decoded, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss:.4f}")

# 5. Extract Embeddings
print("Extracting embeddings for clustering...")
model.eval()
embeddings = []

with torch.no_grad():
    for images, _ in data_loader:
        images = images.view(images.size(0), -1).to(device)
        encoded, _ = model(images)
        embeddings.append(encoded.cpu().numpy())

features = np.concatenate(embeddings)

# 6. K-Means Clustering
print("Running K-Means clustering...")
kmeans = KMeans(n_clusters=10, random_state=42)
cluster_labels = kmeans.fit_predict(features)

# 7. Evaluation
print("Evaluating clustering...")
sil_score = silhouette_score(features, cluster_labels)
db_index = davies_bouldin_score(features, cluster_labels)
print(f"Silhouette Score: {sil_score:.4f}")
print(f"Davies-Bouldin Index: {db_index:.4f}")

# 8. Visualization with t-SNE
print("Generating t-SNE visualization...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_result = tsne.fit_transform(features)

plt.figure(figsize=(10, 8))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=cluster_labels, cmap='tab10', s=10)
plt.title("t-SNE Clustering Visualization (MNIST)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.colorbar()
plt.tight_layout()
plt.show()
