import argparse
import hashlib
import itertools
import math
import os
import random
from pathlib import Path
from typing import Optional
from einops import rearrange
from collections import OrderedDict
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
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
from datasets.ubc_deepfashion_dataset import DreamPoseDataset
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
    clip_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").cuda()
    clip_encoder.requires_grad_(False)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load models and create wrapper for stable diffusion
    vae = AutoencoderKL.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                subfolder="vae",
                revision="ebb811dd71cdc38a204ecbdd6ac5d580f529fd8c"
            )

    # Load pretrained UNet layers
    unet = get_unet3(args.pretrained_model_name_or_path, args.revision, resolution=args.resolution)
    #unet = get_unet('CompVis/stable-diffusion-v1-4')

    if args.custom_chkpt is not None:
        print("Loading ", args.custom_chkpt)
        unet_state_dict = torch.load(args.custom_chkpt)
        new_state_dict = OrderedDict()
        for k, v in unet_state_dict.items():
            name = k[7:] if k[:7] == 'module' else k 
            new_state_dict[name] = v
        unet.load_state_dict(new_state_dict)
        unet = unet.cuda()

    # Embedding adapter
    adapter = Embedding_Adapter(input_nc=1280, output_nc=1280)

    if args.custom_chkpt is not None:
        adapter_chkpt = args.custom_chkpt.replace('unet_epoch', 'adapter')
        print("Loading ", adapter_chkpt)
        adapter_state_dict = torch.load(adapter_chkpt)
        new_state_dict = OrderedDict()
        for k, v in adapter_state_dict.items():
            name = k[7:] if k[:7] == 'module' else k 
            new_state_dict[name] = v
        adapter.load_state_dict(new_state_dict)
        adapter = adapter.cuda()

    #adapter.requires_grad_(True)

    vae.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        adapter.enable_gradient_checkpointing()

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
        itertools.chain(unet.parameters(), adapter.parameters(),)
    )

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=args.revision,
        )
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
        )

    train_dataset = DreamPoseDataset(
        instance_data_root=args.instance_data_dir,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        size=args.resolution,
        center_crop=args.center_crop,
    )

    def collate_fn(examples):
        frame_i = [example["frame_i"] for example in examples]
        frame_j = [example["frame_j"] for example in examples]
        poses = [example["pose_j"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if args.with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            frame_i += [example["class_frame_i"] for example in examples]
            frame_j += [example["class_frame_j"] for example in examples]
            poses += [example["class_pose_j"] for example in examples]

        frame_i = torch.cat(frame_i, 0)
        frame_j = torch.cat(frame_j, 0)
        poses = torch.cat(poses, 0)

        # Dropout
        p = random.random()
        if p <= args.dropout_rate / 3: # dropout pose
            poses = torch.zeros(poses.shape)
        elif p <= 2*args.dropout_rate / 3: # dropout image
            frame_i = torch.zeros(frame_i.shape)
            #frame_k = torch.zeros(frame_k.shape)
        elif p <= args.dropout_rate: # dropout image and pose
            poses = torch.zeros(poses.shape)
            frame_i = torch.zeros(frame_i.shape)
            #frame_k = torch.zeros(frame_k.shape)

        frame_i = frame_i.to(memory_format=torch.contiguous_format).float()
        frame_j = frame_j.to(memory_format=torch.contiguous_format).float()
        #frame_k = frame_k.to(memory_format=torch.contiguous_format).float()
        poses = poses.to(memory_format=torch.contiguous_format).float()
        #joints = joints.to(memory_format=torch.contiguous_format).float()

        batch = {
            "frame_i": frame_i,
            "frame_j": frame_j,
            #"frame_k": frame_k,
            "poses": poses,
            #"joints_j": joints,
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

    if args.train_text_encoder:
        unet, adapter, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, adapter, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, adapter, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, adapter, optimizer, train_dataloader, lr_scheduler
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the image_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)

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
        unet.train()
        adapter.train()
        first_batch = True
        for step, batch in enumerate(train_dataloader):
            if first_batch and latest_chkpt_step is not None:
                #os.system(f"python test_img2img.py --step {latest_chkpt_step} --strength 0.8")
                first_batch = False
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["frame_j"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Concatenate pose with noise
                _, _, h, w = noisy_latents.shape
                noisy_latents = torch.cat((noisy_latents, F.interpolate(batch['poses'], (h,w))), 1)

                # Get CLIP embeddings
                inputs = clip_processor(images=list(batch['frame_i'].to(latents.device)), return_tensors="pt")
                inputs = {k: v.to(latents.device) for k, v in inputs.items()}
                clip_hidden_states =  clip_encoder(**inputs).last_hidden_state.to(latents.device)

                #print("clip states shape = ", clip_hidden_states.shape)

                # Get VAE embeddings
                #print("frame i shape = ", batch['frame_i'].shape)
                image = batch['frame_i'].to(device=latents.device, dtype=weight_dtype)
                vae_hidden_states = vae.encode(image).latent_dist.sample() * 0.18215
                #print("vae states shape = ", vae_hidden_states.shape)

                encoder_hidden_states = adapter(clip_hidden_states, vae_hidden_states)

                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)

                    # Compute instance loss
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none").mean([1, 2, 3]).mean()

                    # Compute prior loss
                    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss
                else:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters())
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
            if global_step % 10 == 0:
                weights = adapter.linear1.weight.cpu().detach().numpy()
                weights = np.sum(weights, axis=0)
                weights = weights.flatten()
                plt.figure()
                plt.plot(range(len(weights)), weights)
                plt.title(f"VAE Weights = {weights[50:]}")
                #plt.hist(weights)
                writer.add_figure('embedding_weights', plt.gcf(), global_step=global_step)
            #add_image("emebedding_weights", h, global_step=global_step)
            #writer.add_histogram('embedding_weights', weights, global_step=global_step, bins='tensorflow')
            writer.add_scalar("loss/train", loss.detach().item(), global_step)
            if global_step % 50 == 0:
                with torch.no_grad():
                    pred_latents = noisy_latents[:,:4,:,:] - model_pred
                    pred_images = latents2img(pred_latents)
                    noise_viz = latents2img(noisy_latents[:,:4,:,:])
                    target = inputs2img(batch["frame_j"])
                    input_img = inputs2img(batch["frame_i"])
                    middle_pose = visualize_dp(target[0], batch['poses'][0][4:6])

                    pose_viz = []
                    for pose_id in range(0, 5):
                        start, end = 2*pose_id, 2*(pose_id+1)
                        pose = visualize_dp(target[0], batch['poses'][0][start:end])
                        pose_viz.append(pose)

                    pose_viz = np.concatenate(pose_viz, axis=2)
                    frame_viz = np.concatenate([input_img[0], noise_viz[0], middle_pose, pred_images[0], target[0]], axis=2)
                    viz = np.concatenate([frame_viz, pose_viz], axis=1)
                    writer.add_image(f'train/pred_img', viz, global_step=global_step)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

            # save model
            if accelerator.is_main_process and global_step % 500 == 0:
                pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    #adapter=accelerator.unwrap_model(adapter),
                    unet=accelerator.unwrap_model(unet),
                    tokenizer=tokenizer,
                    image_encoder=accelerator.unwrap_model(clip_encoder),
                    clip_processor=accelerator.unwrap_model(clip_processor),
                    revision=args.revision,
                )
                pipeline.save_pretrained(os.path.join(args.output_dir, f'checkpoint-{epoch}'))
                model_path = args.output_dir+f'/unet_epoch_{epoch}.pth'
                torch.save(unet.state_dict(), model_path)
                adapter_path = args.output_dir+f'/adapter_{epoch}.pth'
                torch.save(adapter.state_dict(), adapter_path)

        accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)