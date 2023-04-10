from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os, cv2, glob

''' 
    - Passes 5 consecutive input poses per sample 
    - Ensures at least one pair of consecutive frames per batch
'''
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
        p_jitter=0.9,
        n_poses=5
    ):
        self.size = (640, 512)
        self.center_crop = center_crop
        self.train = train
        self.n_poses = n_poses

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        # Load UBC Fashion Dataset
        self.instance_images_path = glob.glob('../UBC_Fashion_Dataset/train-frames/*/*png')
        self.instance_images_path = [p for p in self.instance_images_path if os.path.exists(p.replace('.png', '_densepose.npy'))]
        len1 = len(self.instance_images_path)
        
        # Load Deep Fashion Dataset
        self.instance_images_path.extend([path for path in glob.glob('../Deep_Fashion_Dataset/img_highres/*/*/*/*.jpg') \
                                            if os.path.exists(path.replace('.jpg', '_densepose.npy'))])

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
                #transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
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
        h2, w2 = self.size, self.size
        resized_pose = np.zeros((h2, w2))
        x_vals = np.where(pose != 0)[0]
        y_vals = np.where(pose != 0)[1]
        for (x, y) in list(zip(x_vals, y_vals)):
            # find new coordinates
            x2, y2 = int(x * h2 / h1), int(y * w2 / w1) 
            resized_pose[x2, y2] = pose[x, y]
        return resized_pose
        
    # return two consecutive frames per call
    def __getitem__(self, index):
        example = {}

        '''

        Prepare frame #1
        
        '''
        # load frame i
        frame_path = self.instance_images_path[index % self.num_instance_images]
        instance_image = Image.open(frame_path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["frame_i"] = self.image_transforms(instance_image)

        # Get additional frames in this folder
        sample_folder = frame_path.replace(os.path.basename(frame_path), '')
        samples = [path for path in glob.glob(sample_folder+'/*') if 'npy' not in path]
        samples = [path for path in samples if os.path.exists(path.replace('.jpg', '_densepose.npy').replace('.png', '_densepose.npy'))]

        if 'Deep_Fashion' in frame_path:
            idx = os.path.basename(frame_path).split('_')[0]
            samples = [s for s in samples if os.path.basename(s).split('_')[0] == idx]
            #print("Frame Path = ", frame_path)
            #print("Sampels = ", samples)

        frame_j_path = samples[np.random.choice(range(len(samples)))]
        pose_j_path = frame_j_path.replace('.jpg', '_densepose.npy')

        # load frame j
        instance_image = Image.open(frame_j_path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["frame_j"] = self.image_transforms(instance_image)

        # Load 5 poses surrounding j
        _, h, w = example["frame_i"].shape
        poses = []
        idx1= int(self.n_poses // 2)
        idx2 = self.n_poses - idx1
        for pose_number in range(5):
            dp_path = frame_j_path.replace('.jpg', '_densepose.npy').replace('.png', '_densepose.npy')
            dp_i = F.interpolate(torch.from_numpy(np.load(dp_path, allow_pickle=True).astype('float32')).unsqueeze(0), (h, w), mode='bilinear').squeeze(0)
            poses.append(self.tensor_transforms(dp_i))

        example["pose_j"] = torch.cat(poses, 0)

        '''

        Prepare frame #2
        
        '''
        new_frame_path = samples[np.random.choice(range(len(samples)))]
        frame_path = new_frame_path

        # load frame i
        instance_image = Image.open(frame_path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["frame_i"] = torch.stack((example["frame_i"], self.image_transforms(instance_image)), 0)

        assert example["frame_i"].shape == (2, 3, 640, 512)

        # Load frame j
        frame_j_path = samples[np.random.choice(range(len(samples)))]
        instance_image = Image.open(frame_j_path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["frame_j"] = torch.stack((example['frame_j'], self.image_transforms(instance_image)), 0)

        # Load 5 poses surrounding j
        poses = []
        for pose_number in range(5):
            dp_path = frame_j_path.replace('.jpg', '_densepose.npy').replace('.png', '_densepose.npy')
            dp_i = F.interpolate(torch.from_numpy(np.load(dp_path, allow_pickle=True).astype('float32')).unsqueeze(0), (h, w), mode='bilinear').squeeze(0)
            poses.append(self.tensor_transforms(dp_i))

        poses = torch.cat(poses, 0)
        example["pose_j"] = torch.stack((example["pose_j"], poses), 0)

        #print(example["frame_i"].shape, example["frame_j"].shape, example["pose_j"].shape)
        return example
