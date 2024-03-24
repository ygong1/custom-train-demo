#! python

import time
import time
import warnings
from typing import Any, Dict, List, Optional, Union

import torch
from composer import Trainer
from composer.core.callback import Callback
from composer.loggers import MosaicMLLogger, MLFlowLogger
from composer.loggers.mosaicml_logger import (MOSAICML_ACCESS_TOKEN_ENV_VAR,
                                              MOSAICML_PLATFORM_ENV_VAR)
from composer.utils import dist, get_device, reproducibility
import torch
import torch.utils.data
from dataclasses import dataclass
import os

import composer
import matplotlib.pyplot as plt
import math

from torchvision import datasets, transforms
from composer.loggers import InMemoryLogger
from models.model import ResNetCIFAR
from composer.models import ComposerClassifier
from llmfoundry.composerpatch import MLFlowLogger
import logging

log = logging.getLogger(__name__)


# examples comes from https://docs.mosaicml.com/projects/composer/en/latest/examples/getting_started.html

def create_dataloader(dataset, rank, world_size, batch_size):
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, # shuffle=True
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, num_workers=1, drop_last=True
    )

@dataclass
class MyConfig:
    name: str
    seed: int = 42
    dist_timeout: Union[int, float] = 600.0
    global_train_batch_size: int = 256
    device_train_microbatch_size: int = 16 # the batch size for each GPU device

    
   

    def dist_init(self):
        self.world_size = dist.get_world_size()
        self.node_rank = dist.get_node_rank()
        self.global_rank = dist.get_global_rank()
        self.local_rank = dist.get_local_rank()
        self.experiment_name: str = f"/Users/yu.gong@databricks.com/{self.name}"
    

        self.device_train_batch_size = self.global_train_batch_size // self.world_size
        self.device_train_grad_accum = math.ceil(self.device_train_batch_size /
                                              self.device_train_microbatch_size)

def wait_for_file(filename, timeout=60):
    """
    Wait for a file to exist.
    
    Parameters:
    - filename: Path to the file to wait for.
    - timeout: How long to wait for the file, in seconds.
    """
    start_time = time.time()
    while not os.path.exists(filename):
        log.info(f"Waiting for file {filename} to appear.")
        time.sleep(3)  # Sleep for a short time to wait before checking again.
        if time.time() - start_time > timeout:
            raise TimeoutError(f"File {filename} did not appear within {timeout} seconds.")
    time.sleep(3)  # Sleep for a short time to ensure the file is ready.
    log.info(f"File {filename} is ready.")


def main(cfg: MyConfig) -> Trainer:
     reproducibility.seed_all(cfg.seed)

     # the wrapper function for the setting up the distributed training environment
     # e.g. torch.distributed.init_process_group and the envrionment variables(e.g NODE_RANK,
     # LOCAL_RANK, WORLD_SIZE, etc)
     dist.initialize_dist(get_device(None), timeout=cfg.dist_timeout)

     cfg.dist_init()


     logging.basicConfig(
            # Example of format string
            # 2022-06-29 11:22:26,152: rank0[822018][MainThread]: INFO: Message here
            format=
            f'%(asctime)s: rank{dist.get_global_rank()}[%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s'
        )
     log.setLevel("INFO")
     from dataclasses import asdict
     log.info(f"config: {json.dumps(asdict(cfg))}")
     log.info(f"world_size: {cfg.world_size}, global_rank: {cfg.global_rank}, global_train_batch_size: {cfg.global_train_batch_size}")

     data_directory = "/tmp/data"

     # Normalization constants
     mean = (0.507, 0.487, 0.441)
     std = (0.267, 0.256, 0.276)
     cifar10_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

     if cfg.local_rank == 0:
          logging.info("Downloading CIFAR-10 dataset")
          train_dataset = datasets.CIFAR10(data_directory, train=True, download=True, transform=cifar10_transforms)
          test_dataset = datasets.CIFAR10(data_directory, train=False, download=True, transform=cifar10_transforms)
     else:
          wait_for_file(os.path.join(data_directory, "cifar-10-batches-py/batches.meta"))
          train_dataset = datasets.CIFAR10(data_directory, train=True, download=False, transform=cifar10_transforms)
          test_dataset = datasets.CIFAR10(data_directory, train=False, download=False, transform=cifar10_transforms)
     

     # Our train and test dataloaders are PyTorch DataLoader objects!
     train_dataloader = create_dataloader(train_dataset, cfg.global_rank, cfg.world_size, cfg.global_train_batch_size)
     test_dataloader = create_dataloader(test_dataset, cfg.global_rank, cfg.world_size, cfg.global_train_batch_size)
     

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
        alpha_f=1.0)

     loggers = [InMemoryLogger()]
     if os.environ.get(MOSAICML_PLATFORM_ENV_VAR, 'false').lower(
        ) == 'true' and os.environ.get(MOSAICML_ACCESS_TOKEN_ENV_VAR):
            # Adds mosaicml logger to composer if the run was sent from Mosaic platform, access token is set, and mosaic logger wasn't previously added
            mosaicml_logger = MosaicMLLogger()
            loggers.append(mosaicml_logger)
     databricks_logger = MLFlowLogger.MLFlowLogger(experiment_name=cfg.experiment_name, 
                                                   tracking_uri='databricks', synchronous=False, log_system_metrics=True)
     loggers.append(databricks_logger)
    
  
     train_epochs = "1ep" # Train for 3 epochs because we're assuming Colab environment and hardware
     device = "gpu" if torch.cuda.is_available() else "cpu" # select the device

     trainer = composer.trainer.Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=test_dataloader,
        max_duration=train_epochs,
        optimizers=optimizer,
        schedulers=lr_scheduler,
        device=device,
        loggers=loggers,
    )

     start_time = time.perf_counter()
     log.info(f"Starting training: start_time={start_time}")
     trainer.fit() # <-- Your training loop in action!
     end_time = time.perf_counter()
     log.info(f"It took {end_time - start_time:0.4f} seconds to train")

import sys
import json

if __name__ == '__main__':
    json_str = sys.argv[1]
    cfg = MyConfig(**json.loads(json_str))
    main(cfg)