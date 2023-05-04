from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os, cv2, glob

class DreamPoseDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
        train=True,
        p_jitter=0.9
    ):
        self.size = (640, 512)
        self.center_crop = center_crop
        self.train = train

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        # Load UBC Fashion Dataset
        self.instance_images_path = glob.glob(instance_data_root+'/*png')

        self.num_instance_images = len(self.instance_images_path)
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )

        self.tensor_transforms = transforms.Compose(
            [
            ]
        )

    def __len__(self):
        return self._length

    # resize sparse uv flow to size
    def resize_pose(self, pose):
        h1, w1 = pose.shape
        h2, w2 = self.size, self.size
        resized_pose = np.zeros((h2, w2))
        x_vals = np.where(pose != 0)[0]
        y_vals = np.where(pose != 0)[1]
        for (x, y) in list(zip(x_vals, y_vals)):
            # find new coordinates
            x2, y2 = int(x * h2 / h1), int(y * w2 / w1) 
            resized_pose[x2, y2] = pose[x, y]
        return resized_pose
        
    def __getitem__(self, index):
        example = {}

        frame_path = self.instance_images_path[index % self.num_instance_images]
        frame_folder = frame_path.replace(os.path.basename(frame_path), '')
        #frame_number = int(os.path.basename(frame_path).split('frame_')[-1].replace('.png', ''))

        # load frame i
        instance_image = Image.open(frame_path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        example["frame_i"] = self.image_transforms(instance_image)
        example["frame_prev"] = self.image_transforms(instance_image)

        assert example["frame_i"].shape == (3, 640, 512)

        # Select other frame in this folder
        frame_paths = glob.glob(frame_folder+'/*png')
        frame_paths = [p for p in frame_paths if os.path.exists(p.replace('.png', '_densepose.npy'))]
        frame_j_path = np.random.choice(frame_paths)

        # load frame j
        frame_j_path = np.random.choice(frame_paths)
        instance_image = Image.open(frame_j_path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["frame_j"] = self.image_transforms(instance_image)


        # construct 5 input poses
        poses = []
        h, w = 640, 512
        for pose_number in range(5):
            dp_path = frame_j_path.replace('.png', '_densepose.npy')
            dp_i = F.interpolate(torch.from_numpy(np.load(dp_path).astype('float32')).unsqueeze(0), (h, w), mode='bilinear').squeeze(0)
            poses.append(self.tensor_transforms(dp_i))
        input_pose = torch.cat(poses, 0)
        example["pose_j"] = input_pose
        
        ''' Data Augmentation '''
        key_frame = example["frame_i"] 
        frame = example["frame_j"]
        prev_frame = example["frame_prev"]

        #dp = transforms.ToPILImage()(dp)

        # Get random transforms to target 70% of the time
        p = np.random.randint(0, 100)
        if p < 70:
            ang = np.random.randint(-15, 15) # rotation angle
            distort = np.random.rand(0, 1)
            top, left = np.random.randint(0, 25), np.random.randint(0, 25)
            h_ = np.random.randint(self.size[0]-25, self.size[0]-top)
            w_ = int(h_ / h * w)

            t = transforms.Compose([transforms.ToPILImage(),\
                                    transforms.Resize((h,w), interpolation=transforms.InterpolationMode.BILINEAR), \
                                    transforms.ToTensor(),\
                                    ])

            # Apply transforms
            frame = transforms.functional.crop(frame, top, left, h_, w_) # random crop

            example["frame_j"] = t(frame)

            for pose_id in range(5):
                start, end = 2*pose_id, 2*pose_id+2
                # convert dense pose to PIL image
                dp = example['pose_j'][start:end]
                c, h, w = dp.shape
                dp = torch.cat((dp, torch.zeros(1, h, w)), 0)
                dp = transforms.functional.crop(dp, top, left, h_, w_) # random crop
                dp = t(dp)[0:2] # Remove extra channel from input pose
                example["pose_j"][start:end] = dp.clone()

            # slightly perturb transforms to previous frame, to prevent copy/paste
            top += np.random.randint(0, 5)
            left += np.random.randint(0, 5)
            h_ += np.random.randint(0, 5)
            w_ += np.random.randint(0, 5)
            prev_frame = transforms.functional.crop(prev_frame, top, left, h_, w_) # random crop
            example["frame_prev"] = t(prev_frame)
        else:
            # slightly perturb transforms to previous frame, to prevent copy/paste
            top, left = np.random.randint(0, 5), np.random.randint(0, 5)
            h_ = np.random.randint(self.size[0]-5, self.size[0]-top)
            w_ = int(h_ / h * w)

            t = transforms.Compose([transforms.ToPILImage(),\
                                    transforms.Resize((h,w), interpolation=transforms.InterpolationMode.BILINEAR), \
                                    transforms.ToTensor(),\
                                    ])

            prev_frame = transforms.functional.crop(prev_frame, top, left, h_, w_) # random crop
            example["frame_prev"] = t(prev_frame)

            for pose_id in range(5):
                start, end = 2*pose_id, 2*pose_id+2
                dp = example['pose_j'][start:end]
                example["pose_j"][start:end] = dp.clone()

        return example
