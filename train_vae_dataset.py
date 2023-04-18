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
        self.instance_images_path = [path for path in glob.glob(instance_data_root+'/*/*/*') if 'frame_i.png' in path]

        if len(self.instance_images_path) == 0:
            self.instance_images_path = [path for path in glob.glob(instance_data_root+'/*') if 'png' in path]

        len1 = len(self.instance_images_path)
        # Load Deep Fashion Dataset
        #self.instance_images_path.extend([path for path in glob.glob('../Deep_Fashion_Dataset/img_highres/*/*/*/*.jpg') \
        #                                    if os.path.exists(path.replace('.jpg', '_densepose.npy'))])

        len2 = len(self.instance_images_path)
        print(f"Train Dataset: {len1} UBC Fashion images, {len2-len1} Deep Fashion images.")

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
                #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.3, hue=0.3),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.tensor_transforms = transforms.Compose(
            [
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    # resize sparse uv flow to size
    def resize_pose(self, pose):
        h1, w1 = pose.shape
        h2, w2 = self.size[0], self.size[1]
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

        # load frame j
        frame_path = frame_path.replace('frame_i', 'frame_j')
        instance_image = Image.open(frame_path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        frame_j = instance_image
        frame_j = frame_j.resize((self.size[1], self.size[0]))

        # Load pose j
        h, w = self.size[0], self.size[1]
        dp_path = self.instance_images_path[index % self.num_instance_images].replace('frame_i', 'frame_j').replace('.png', '_densepose.npy')
        dp_j = F.interpolate(torch.from_numpy(np.load(dp_path, allow_pickle=True).astype('float32')).unsqueeze(0), (h, w), mode='bilinear').squeeze(0)
        
        # Load joints j
        #pose_path = self.instance_images_path[index % self.num_instance_images].replace('frame', 'pose').replace('.png', '_refined.npy')
        #pose = np.load(pose_path).astype('float32')
        #pose = self.resize_pose(pose / 32).astype('float32')
        #joints_j = torch.from_numpy(pose).unsqueeze(0) 

        # Apply random crops
        max_crop = int(0.1*min(frame_j.size[0], frame_j.size[1]))
        top, left = np.random.randint(0, max_crop), np.random.randint(0, max_crop)
        h_ = np.random.randint(self.size[0]-max_crop, self.size[0]-top)
        w_ = int(h_ / h * w)
        #print(self.size[0]-max_crop, self.size[0]-top, h_, w_)
        frame_j = transforms.functional.crop(frame_j, top, left, h_, w_) # random crop
        dp_j = transforms.functional.crop(dp_j, top, left, h_, w_) # random crop
        #joints_j = transforms.functional.crop(joints_j, top, left, h_, w_) # random crop

        # Apply resize and normalization
        example["frame_j"] = self.image_transforms(frame_j)
        dp_j = self.tensor_transforms(dp_j)
        example["pose_j"] = F.interpolate(dp_j.unsqueeze(0), (h, w), mode='bilinear').squeeze(0)

        #joints_j = self.resize_pose(joints_j[0].numpy())
        #example["joints_j"] = torch.from_numpy(joints_j).unsqueeze(0) 

        return example
