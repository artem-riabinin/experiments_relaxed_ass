#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import random

# Set parameters
learning_rate = 0.05
batch_size = 128
epochs = 300
input_dim = 784  # 28x28 images from Fashion-MNIST
hidden_dim = 64
output_dim = 10  # 10 classes in Fashion-MNIST
c_1 = 1
c_11 = 0.01
num_seeds = 3  # Number of different seeds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def predict(model, inputs):
    with torch.no_grad():
        outputs = model(inputs)
    return outputs

# Load dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Store results for different seeds
all_losses = []
all_gradients = []
all_inner_products = []
all_inner_products2 = []

# Training function
def train(seed):
    # Fix seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize model, loss function, and optimizer
    model = MLP(input_dim, hidden_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    epoch_losses = []
    epoch_gradients = []
    epoch_params = []
    losses = []
    gradients_norm = []
    inputs_saved = []
    targets_saved = []

    # Training loop
    for epoch in range(epochs):
        print('epoch', epoch)
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.view(-1, 28 * 28).to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            if batch_idx == len(train_loader) - 1:
                gradients = [param.grad.clone() for param in model.parameters()]
                epoch_gradients.append(gradients)
                epoch_losses.append(loss.item())
                params = [param.clone().detach() for param in model.parameters()]
                epoch_params.append(params)
                inputs_saved.append(inputs)
                targets_saved.append(targets)

            optimizer.step()

        # Compute loss and gradient norm
        #total_loss = 0.0
        #num_samples = len(train_loader.dataset)
        #for inputs, targets in train_loader:
        #    inputs, targets = inputs.view(-1, 28 * 28).to(device), targets.to(device)
        #    outputs = model(inputs)
        #    total_loss += criterion(outputs, targets) * inputs.shape[0]
        #total_loss /= num_samples
        #grads = torch.autograd.grad(total_loss, model.parameters(), create_graph=False)
        #losses.append(total_loss.detach())
        #gradients_norm.append(torch.norm(torch.cat([p.detach().view(-1) for p in grads])))

        #print(f"Seed {seed}, Epoch {epoch+1}/{epochs}, Loss: {losses[-1]}, Gradient Norm: {gradients_norm[-1]}")

    # Compute inner products
    x_star = [param.clone().detach() for param in epoch_params[-1]]
    inner_products = []
    inner_products2 = []
    for epoch in range(epochs):
        grad_vector = torch.cat([g.view(-1) for g in epoch_gradients[epoch]])
        param_vector = torch.cat([p.view(-1) for p in epoch_params[epoch]])
        x_star_vector = torch.cat([x.view(-1) for x in x_star])
        inner_product = torch.dot(grad_vector, param_vector - x_star_vector) - c_1 * epoch_losses[epoch] + c_1 * criterion(predict(model, inputs_saved[epoch]), targets_saved[epoch])
        print(- c_1 * epoch_losses[epoch] + c_1 * criterion(predict(model, inputs_saved[epoch]), targets_saved[epoch]))
        inner_products.append(inner_product.item())
        inner_product2 = torch.dot(grad_vector, param_vector - x_star_vector) - c_11 * torch.norm(grad_vector) ** 2
        inner_products2.append(inner_product2.item())

    return losses, gradients_norm, inner_products, inner_products2

# Run training for different seeds
for seed in range(num_seeds):
    losses, gradients_norm, inner_products, inner_products2 = train(seed)
    all_losses.append(losses)
    all_gradients.append(gradients_norm)
    all_inner_products.append(inner_products)
    all_inner_products2.append(inner_products2)


# In[ ]:


# Convert to NumPy arrays
all_losses = np.array(all_losses)
all_gradients = np.array(all_gradients)
all_inner_products = np.array(all_inner_products)
all_inner_products2 = np.array(all_inner_products2)


#all_losses = np.array([[loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in inner_list] for inner_list in all_losses])
#all_gradients = np.array([[grad.cpu().numpy() if isinstance(grad, torch.Tensor) else grad for grad in inner_list] for inner_list in all_gradients])
#all_inner_products = np.array([[inner_product.cpu().numpy() if isinstance(inner_product, torch.Tensor) else inner_product for inner_product in inner_list] for inner_list in all_inner_products])
#all_inner_products2 = np.array([[inner_product2.cpu().numpy() if isinstance(inner_product2, torch.Tensor) else inner_product2 for inner_product2 in inner_list] for inner_list in all_inner_products2])


# Compute statistics
mean_inner_products = np.mean(all_inner_products, axis=0)
min_inner_products = np.min(all_inner_products, axis=0)
max_inner_products = np.max(all_inner_products, axis=0)

mean_inner_products2 = np.mean(all_inner_products2, axis=0)
min_inner_products2 = np.min(all_inner_products2, axis=0)
max_inner_products2 = np.max(all_inner_products2, axis=0)

# Plotting the results
plt.style.use('default')
fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # Corrected subplot setup

# Plot 1: First case
axs[0].plot(mean_inner_products, label='Mean', linewidth=1.5)
axs[0].fill_between(range(len(mean_inner_products)), min_inner_products, max_inner_products,
                    alpha=0.3, label='Min-Max')
axs[0].set_title(r'$\left\langle \nabla f_{\xi}(x^k), x^k - x^K \right\rangle - c_1 \left(f_{\xi}(x^k) - f_{\xi}(x^K)\right)$, $c_1=1$')
axs[0].set_xlabel('Epoch $k$')
axs[0].set_ylabel(r'$\left\langle \nabla f_{\xi}(x^k), x^k - x^K \right\rangle - c_1 \left(f_{\xi}(x^k) - f_{\xi}(x^K)\right)$')
axs[0].legend(loc='upper right')
axs[0].grid(True)

# Plot 2: Second case
axs[1].plot(mean_inner_products2, label='Mean', linewidth=1.5)
axs[1].fill_between(range(len(mean_inner_products2)), min_inner_products2, max_inner_products2,
                    alpha=0.3, label='Min-Max')
axs[1].set_title(r'$\left\langle \nabla f_{\xi}(x^k), x^k - x^K \right\rangle - c_1 \|\nabla f_{\xi}(x^k)\|^2$, $c_1=0.01$')
axs[1].set_xlabel('Epoch $k$')
axs[1].set_ylabel(r'$\left\langle \nabla f_{\xi}(x^k), x^k - x^K \right\rangle - c_1 \|\nabla f_{\xi}(x^k)\|^2$')
axs[1].legend(loc='upper right')
axs[1].grid(True)

# Layout and saving
plt.tight_layout()
plt.savefig("plot2.pdf", format="pdf", dpi=300, bbox_inches="tight")
plt.show()

# Find c_2 (for both cases)
max_c2 = -np.min(all_inner_products)
print("Max c_2 for the first case:", max_c2)
max_c2_2 = -np.min(all_inner_products2)
print("Max c_2 for the second case:", max_c2_2)

# Find c_2
max_c2 = -np.min(all_inner_products)
print("Max c_2 for the first case:", max_c2)
max_c2_2 = -np.min(all_inner_products2)
print("Max c_2 for the second case:", max_c2_2)

