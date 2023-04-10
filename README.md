# DreamPose
Official implementation of "DreamPose: Fashion Image-to-Video Synthesis via Stable Diffusion"

## Demo

You can generate a video using DreamPose using our pretrained models.

1. Download pretrained models and store them into the "checkpoints" folder.
2. Run demo.py using the command below:
    ```
    python test.py --epoch 20 --folder checkpoints --pose_folder demo/sample/poses  --key_frame_path demo/sample/key_frame.png --s1 8 --s2 3 --n_steps 100 --output_dir results
    ```

## Training


## Testing

You can finetune the base model on your own image and generate videos using a pose sequence.

1. Data preparation

  demo/sample/train-frame
    

2. Finetune the UNet

    ```
    accelerate launch finetune-unet.py --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" --instance_data_dir=demo/sample/train --output_dir=demo/custom-chkpts --resolution=512 --train_batch_size=1 --gradient_accumulation_steps=1 --learning_rate=1e-5 --num_train_epochs=500 --dropout_rate=0.0 --custom_chkpt=checkpoints/unet_epoch_20.pth --revision "ebb811dd71cdc38a204ecbdd6ac5d580f529fd8c"
    ```

3. Finetune the VAE decoder

    ```
    accelerate launch --num_processes=1 finetune-vae.py --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  --instance_data_dir=demo/sample/train --output_dir=demo/custom-chkpts --instance_prompt="" --resolution=512  --train_batch_size=4 --gradient_accumulation_steps=4 --learning_rate=5e-5 --num_train_epochs=1500 --run_name finetuning/ubc-vae --revision "ebb811dd71cdc38a204ecbdd6ac5d580f529fd8c"
    ```

4. Generate predictions

    ```
    python test.py --epoch 20 --folder demo/custom-chkpts --pose_folder demo/sample/poses  --key_frame_path demo/sample/key_frame.png --s1 8 --s2 3 --n_steps 100 --output_dir results
    ```
    
