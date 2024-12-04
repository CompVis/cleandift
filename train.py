import argparse
import hydra
import logging
import os
import torch
from typing import Optional
from omegaconf import DictConfig, OmegaConf
from src.utils import set_seed, dict_to
from transformers import get_scheduler
from tqdm.auto import tqdm


@hydra.main(config_path="configs", config_name="sd15_feature_extractor", version_base="1.1")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    set_seed(cfg.seed)
    logger = logging.getLogger(f"{__name__}")
    device = torch.device("cuda:0")

    # Load model
    cfg = hydra.utils.instantiate(cfg)
    model = cfg.model.to(device)
    model.train()

    data = cfg.data
    dataloader_train = data.train_dataloader()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    lr_scheduler = get_scheduler(
        name=cfg.lr_scheduler.name,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_scheduler.num_warmup_steps,
        num_training_steps=cfg.lr_scheduler.num_training_steps,
        scheduler_specific_kwargs=OmegaConf.to_container(cfg.lr_scheduler.scheduler_specific_kwargs),
    )

    i_epoch = -1
    stop = False
    max_steps: Optional[int] = cfg.max_steps

    val_freq: Optional[int] = cfg.val_freq
    if not val_freq is None:
        dataloader_val = data.val_dataloader()
    max_val_steps: Optional[int] = cfg.max_val_steps
    checkpoint_freq: Optional[int] = cfg.checkpoint_freq
    checkpoint_dir: str = cfg.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    grad_accum_steps = cfg.grad_accum_steps    
    print(f"grad_accum_steps={grad_accum_steps}")

    step = 0

    while not stop:  # Epochs
        i_epoch += 1
        for batch in (
            pbar := tqdm(dataloader_train, desc=f"Optimizing (Epoch {i_epoch + 1})")
        ):  
            loss_sum = 0
            for accum_step in range(grad_accum_steps):
                losses = model(**dict_to(batch, device=device))
                loss = sum(v.mean() for v in losses.values())
                loss.backward()
                loss_sum += float(loss.detach().item())
                pbar.set_postfix({ 'loss': loss_sum / (accum_step + 1) })
        
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if not val_freq is None and step % val_freq == 0:
                model.eval()

                with torch.no_grad():
                    val_losses_accumulated = []
                    for i, val_batch in enumerate(
                        tqdm(dataloader_val, desc=f"Validating", total=max_val_steps)
                    ):
                        val_losses = model(**dict_to(val_batch, device=device))
                        val_loss = sum(v.mean() for v in val_losses.values())
                        val_losses_accumulated.append((val_loss).cpu().item())

                        if max_val_steps is not None and i + 1 >= max_val_steps:
                            break

                    val_loss = sum(val_losses_accumulated) / len(val_losses_accumulated)
                    logger.info(f"Validation loss: {val_loss}")

                # put model into train mode
                model.train()

            if not checkpoint_freq is None and (step + 1) % checkpoint_freq == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"step_{(step + 1)}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

            if not max_steps is None and step == max_steps:
                stop = True
                break

            step += 1


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
    main()
