import time

import torch
import torch.utils.data

import composer
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from composer.loggers import InMemoryLogger

# examples comes from https://docs.mosaicml.com/projects/composer/en/latest/examples/getting_started.html

torch.manual_seed(42) # For replicability


data_directory = "./data"

# Normalization constants
mean = (0.507, 0.487, 0.441)
std = (0.267, 0.256, 0.276)

batch_size = 1024

cifar10_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

train_dataset = datasets.CIFAR10(data_directory, train=True, download=True, transform=cifar10_transforms)
test_dataset = datasets.CIFAR10(data_directory, train=False, download=True, transform=cifar10_transforms)

# Our train and test dataloaders are PyTorch DataLoader objects!
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = ComposerClassifier(module=ResNetCIFAR(), num_classes=10)

optimizer = composer.optim.DecoupledSGDW(
    model.parameters(), # Model parameters to update
    lr=0.05, # Peak learning rate
    momentum=0.9,
    weight_decay=2.0e-3 # If this looks large, it's because its not scaled by the LR as in non-decoupled weight decay
)

lr_scheduler = composer.optim.LinearWithWarmupScheduler(
    t_warmup="1ep", # Warm up over 1 epoch
    alpha_i=1.0, # Flat LR schedule achieved by having alpha_i == alpha_f
    alpha_f=1.0
)

logger_for_baseline = InMemoryLogger()

train_epochs = "3ep" # Train for 3 epochs because we're assuming Colab environment and hardware
device = "gpu" if torch.cuda.is_available() else "cpu" # select the device

trainer = composer.trainer.Trainer(
    model=model,
    train_dataloader=train_dataloader,
    eval_dataloader=test_dataloader,
    max_duration=train_epochs,
    optimizers=optimizer,
    schedulers=lr_scheduler,
    device=device,
    loggers=logger_for_baseline,
)


start_time = time.perf_counter()
trainer.fit() # <-- Your training loop in action!
end_time = time.perf_counter()
print(f"It took {end_time - start_time:0.4f} seconds to train")