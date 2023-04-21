import argparse
import hashlib
import itertools
import math
import os
import random
from pathlib import Path
from typing import Optional
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTokenizer, CLIPProcessor, CLIPVisionModel

from torch.utils.tensorboard import SummaryWriter

logger = get_logger(__name__)

from utils.parse_args import parse_args
from datasets.train_vae_dataset import DreamPoseDataset
from pipelines.dual_encoder_pipeline import StableDiffusionImg2ImgPipeline
from models.unet_dual_encoder import get_unet, Embedding_Adapter

def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    writer = SummaryWriter(f'results/logs/{args.run_name}')

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
    )

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if args.seed is not None:
        set_seed(args.seed)

    # initialize perecpetual loss
    #lpips_loss = lpips.LPIPS(net='vgg').cuda()

    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=args.revision,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")
            pipeline.to(accelerator.device)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load CLIP Image Encoder
    image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load models and create wrapper for stable diffusion
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    )
    # Load pretrained UNet layers
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )
    # Modify input layer & copy pretrain weights
    weights = unet.conv_in.weight.clone()
    unet.conv_in = nn.Conv2d(6, weights.shape[0], kernel_size=3, padding=(1, 1))
    with torch.no_grad():
        unet.conv_in.weight[:, :4] = weights # original weights
        unet.conv_in.weight[:, 4:] = torch.zeros(unet.conv_in.weight[:, 4:].shape) # new weights initialized to zero
    unet.requires_grad_(False)

    # set VAE decoder to be trainable
    # Load VAE Pretrained Model
    if args.custom_chkpt is not None:
        vae_state_dict = torch.load(args.custom_chkpt) #'results/epoch_1/unet.pth'))
        new_state_dict = OrderedDict()
        for k, v in vae_state_dict.items():
            name = k[7:] if k[:7] == 'module' else k # remove `module.`
            new_state_dict[name] = v
        vae.load_state_dict(new_state_dict)
        vae = vae.cuda()

    vae.requires_grad_(False)
    vae_trainable_params = []
    for name, param in vae.named_parameters():
        if 'decoder' in name:
            param.requires_grad = True
            vae_trainable_params.append(param)

    print(f"VAE total params = {len(list(vae.named_parameters()))}, trainable params = {len(vae_trainable_params)}")
    image_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        vae.gradient_checkpointing_enable() # uncomment if training clip model

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = (
        itertools.chain(vae_trainable_params)
    )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")


    train_dataset = DreamPoseDataset(
        instance_data_root=args.instance_data_dir,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        size=args.resolution,
        center_crop=args.center_crop,
    )

    def collate_fn(examples):
        frame_j = [example["frame_j"] for example in examples]
        poses = [example["pose_j"] for example in examples]

        frame_j = torch.stack(frame_j)
        poses = torch.stack(poses)

        frame_j = frame_j.to(memory_format=torch.contiguous_format).float()
        poses = poses.to(memory_format=torch.contiguous_format).float()

        batch = {
            "target_frame": frame_j,
            "poses": poses,
        }
        return batch

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=1
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    unet, vae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, vae, optimizer, train_dataloader, lr_scheduler
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        image_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    def latents2img(latents):
        latents = 1 / 0.18215 * latents
        images = vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.detach().cpu().numpy()
        images = (images * 255).round().astype("uint8")
        return images

    def inputs2img(input):
        target_images = (input / 2 + 0.5).clamp(0, 1)
        target_images = target_images.detach().cpu().numpy()
        target_images = (target_images * 255).round().astype("uint8")
        return target_images

    def visualize_dp(im, dp):
        im = im.transpose((1,2,0))
        hsv = np.zeros(im.shape, dtype=np.uint8)
        hsv[..., 1] = 255

        dp = dp.cpu().detach().numpy()
        mag, ang = cv2.cartToPolar(dp[0], dp[1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        bgr = bgr.transpose((2,0,1))
        return bgr

    latest_chkpt_step = 0
    for epoch in range(args.epoch, args.num_train_epochs):
        vae.train()
        first_batch = True
        for step, batch in enumerate(train_dataloader):
            if first_batch and latest_chkpt_step is not None:
                first_batch = False
            with accelerator.accumulate(vae):
                # Convert images to latent space
                latents = vae.encode(batch["target_frame"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * 0.18215

                latents = 1 / 0.18215 * latents
                pred_images = vae.decode(latents).sample
                pred_images = pred_images.clamp(-1, 1)

                loss = F.mse_loss(pred_images.float(), batch['target_frame'].clamp(-1, 1).float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(vae.parameters())
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            # write to tensorboard
            writer.add_scalar("loss/train", loss.detach().item(), global_step)

            # write to tensorboard
            if global_step % 10 == 0:
                # Draw VAE decoder weights
                weights = vae.decoder.conv_out.weight.cpu().detach().numpy()
                weights = np.sum(weights, axis=0)
                weights = weights.flatten()
                plt.figure()
                plt.plot(range(len(weights)), weights)
                plt.title(f"VAE Decoder Weights = {np.mean(weights)}")
                writer.add_figure('decoder_weights', plt.gcf(), global_step=global_step)

                # Draw VAE encoder weights
                weights = vae.encoder.conv_out.weight.cpu().detach().numpy()
                weights = np.sum(weights, axis=0)
                weights = weights.flatten()
                plt.figure()
                plt.plot(range(len(weights)), weights)
                plt.title(f"Fixed VAE Encoder Weights= {np.mean(weights)}")
                writer.add_figure('encoder_weights', plt.gcf(), global_step=global_step)

            if global_step == 1 or global_step % 50 == 0:
                with torch.no_grad():
                    pred_images = inputs2img(pred_images)
                    target = inputs2img(batch["target_frame"])
                    viz = np.concatenate([pred_images[0], target[0]], axis=2)
                    writer.add_image(f'train/pred_img', viz, global_step=global_step)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

            # save model
            if accelerator.is_main_process and global_step % 500 == 0:
                model_path = args.output_dir+f'/vae_{epoch}.pth'
                torch.save(vae.state_dict(), model_path)

        accelerator.wait_for_everyone()

    # save model
    if accelerator.is_main_process:
        print("Saving final model to ", args.output_dir)
        model_path = args.output_dir+f'/vae_{epoch}.pth'
        torch.save(vae.state_dict(), model_path)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)