import os
from io import BytesIO
from random import choice, random

import cv2
import numpy as np
import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile
from scipy.ndimage import gaussian_filter
from torch.utils.data.sampler import WeightedRandomSampler

from utils.config import CONFIGCLASS

ImageFile.LOAD_TRUNCATED_IMAGES = True


def dataset_folder(root: str, cfg: CONFIGCLASS):
    # For training:
    if cfg.tr_mode:
        #print('tr_mode: training')
        return binary_dataset_v0(root, cfg)
    # For testing:
    if cfg.mode == "binary":
        #print('tr_mode: testing')
        return binary_dataset(root, cfg)
    if cfg.mode == "filename":
        return FileNameDataset(root, cfg)
    raise ValueError("cfg.mode needs to be binary or filename.")


def binary_dataset_v0(root: str, cfg: CONFIGCLASS):
    identity_transform = transforms.Lambda(lambda img: img)

    if cfg.isTrain or cfg.aug_resize:
        rz_func = transforms.Lambda(lambda img: custom_resize(img, cfg))
    else:
        rz_func = identity_transform

    if cfg.isTrain:
        crop_func = transforms.RandomCrop(cfg.cropSize)
    else:
        crop_func = transforms.CenterCrop(cfg.cropSize) if cfg.aug_crop else identity_transform

    if cfg.isTrain and cfg.aug_flip:
        flip_func = transforms.RandomHorizontalFlip()
    else:
        flip_func = identity_transform

    return datasets.ImageFolder(
        root,
        transforms.Compose(
            [
                rz_func,
                transforms.Lambda(lambda img: blur_jpg_augment(img, cfg)),
                crop_func,
                flip_func,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                if cfg.aug_norm
                else identity_transform,
            ]
        )
    )

def binary_dataset(root: str, cfg):
    # Define transformations
    identity_transform = transforms.Lambda(lambda img: img)
    rz_func = transforms.Lambda(lambda img: custom_resize(img, cfg)) if cfg.isTrain or cfg.aug_resize else identity_transform
    crop_func = transforms.RandomCrop(cfg.cropSize) if cfg.isTrain else (transforms.CenterCrop(cfg.cropSize) if cfg.aug_crop else identity_transform)
    flip_func = transforms.RandomHorizontalFlip() if cfg.isTrain and cfg.aug_flip else identity_transform

    transform_pipeline = transforms.Compose([
        rz_func,
        transforms.Lambda(lambda img: blur_jpg_augment(img, cfg)),
        crop_func,
        flip_func,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if cfg.aug_norm else identity_transform,
    ])

    # Load the full dataset
    full_dataset = datasets.ImageFolder(root=root, transform=transform_pipeline)

    # Class names based on cfg.generator
    generator = cfg.generator
    class_names = ['0_real', f'1_fake_{generator}']

    # Mapping from class name to index
    class_to_idx = {class_name: idx for class_name, idx in full_dataset.class_to_idx.items() if class_name in class_names}

    # Find indices of dataset items belonging to the specified classes
    indices = [i for i, (_, class_index) in enumerate(full_dataset.samples) if full_dataset.classes[class_index] in class_names]

    # Create the subset
    subset_dataset = torch.utils.data.Subset(full_dataset, indices)

    return subset_dataset

class FileNameDataset(datasets.ImageFolder):
    def name(self):
        return 'FileNameDataset'

    def __init__(self, opt, root):
        self.opt = opt
        super().__init__(root)

    def __getitem__(self, index):
        # Loading sample
        path, target = self.samples[index]
        return path


def blur_jpg_augment(img: Image.Image, cfg: CONFIGCLASS):
    img: np.ndarray = np.array(img)
    if cfg.isTrain:
        if random() < cfg.blur_prob:
            sig = sample_continuous(cfg.blur_sig)
            gaussian_blur(img, sig)

        if random() < cfg.jpg_prob:
            method = sample_discrete(cfg.jpg_method)
            qual = sample_discrete(cfg.jpg_qual)
            img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def sample_continuous(s: list):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s: list):
    return s[0] if len(s) == 1 else choice(s)


def gaussian_blur(img: np.ndarray, sigma: float):
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)


def cv2_jpg(img: np.ndarray, compress_val: int) -> np.ndarray:
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode(".jpg", img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


def pil_jpg(img: np.ndarray, compress_val: int):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format="jpeg", quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {"cv2": cv2_jpg, "pil": pil_jpg}


def jpeg_from_key(img: np.ndarray, compress_val: int, key: str) -> np.ndarray:
    method = jpeg_dict[key]
    return method(img, compress_val)


rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}
def custom_resize(img: Image.Image, cfg: CONFIGCLASS) -> Image.Image:
    interp = sample_discrete(cfg.rz_interp)
    return TF.resize(img, cfg.loadSize, interpolation=rz_dict[interp])


def get_dataset(cfg: CONFIGCLASS):
    dset_lst = []
    for dataset in cfg.datasets:
        root = os.path.join(cfg.dataset_root, dataset)
        dset = dataset_folder(root, cfg)
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst)


def get_bal_sampler(dataset: torch.utils.data.ConcatDataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1.0 / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights))


def create_dataloader(cfg: CONFIGCLASS):
    shuffle = not cfg.serial_batches if (cfg.isTrain and not cfg.class_bal) else False
    dataset = get_dataset(cfg)
    sampler = get_bal_sampler(dataset) if cfg.class_bal else None

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(cfg.num_workers),
    )
