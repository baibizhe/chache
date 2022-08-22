from monai.utils import first
import os
from types import SimpleNamespace

import matplotlib.pyplot as plt
from monai.data import DataLoader, Dataset, SmartCacheDataset, CacheDataset
from monai.transforms import (
    EnsureTyped,
    Compose,
    LoadImaged,
    RandRotate90d,
    ScaleIntensityd,
    Resized, RandGaussianSmoothd, RandAdjustContrastd, RandGaussianNoised, RandShiftIntensityd,
    ResizeWithPadOrCropd, EnsureChannelFirstd, AsDiscrete, AsDiscreted, RandAxisFlipd, RandFlipd,
)
import numpy as np
from monai.utils import first
from sklearn.model_selection import StratifiedKFold


def get_class_data_loaders(config,fold):
    data_dir = os.path.join("data", "train","image")
    all_images = os.listdir(data_dir)
    image_paths = []
    label_paths = []
    for file_name in all_images:
        file_name_pure = file_name.split(".jpg")[0]
        image_path  =  os.path.join(data_dir,file_name)
        image_paths.append(image_path)
        label_path = os.path.join("data","train","label",file_name_pure+".npy")
        label_paths.append(label_path)


    data_dicts = np.array([
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(image_paths, label_paths)
    ])
    kf = StratifiedKFold(n_splits=5, shuffle=True)

    labels_use_for_split = [np.load(i["label"]) for i in data_dicts]
    labels_use_for_split=np.array(labels_use_for_split).flatten()
    print(np.bincount(labels_use_for_split[0:1582]))

    for idx, (train_idx, val_idx) in enumerate(kf.split(labels_use_for_split, labels_use_for_split)):
        if idx == fold :
            train_files = data_dicts[train_idx]
            val_files = data_dicts[val_idx]
    # splitIndex = int(len(data_dicts) * 0.8)
    # train_files, val_files = data_dicts[:splitIndex], data_dicts[splitIndex:]
    print("train file length {} val_files length{} ".format(len(train_files),len(val_files)))

    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image","label"]),

        ResizeWithPadOrCropd(keys=["image", ],
                             spatial_size = (1296,2304),
                             constant_values = 0
                             ),
        RandAxisFlipd(keys=["image", ],prob=0.3,),
        RandFlipd(keys=["image", ],prob=0.3,),
        # RandGibbsNoised(keys=["image", ],prob=0.3,),
        # AsDiscreted(keys=["label", ],argmax=True, to_onehot=3, threshold=0.5),
        EnsureTyped(keys=["image", "label"])
    ])
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image","label"]),
        ResizeWithPadOrCropd(keys=["image", ],
                             spatial_size=(1296, 2304),
                             constant_values=0
                             ),
        # AsDiscreted(keys=["label", ], argmax=True, to_onehot=3, threshold=0.5),

        EnsureTyped(keys=["image", "label"])
    ])

    train_ds = CacheDataset(
        data=train_files, transform=train_transforms,cache_num=1000,hash_as_key=True
    )
    # train_ds = Dataset(data=train_files, transform=train_transforms)

    train_loader = DataLoader(train_ds, batch_size=30, shuffle=True, num_workers=0)
    #
    val_ds = Dataset(
        data=val_files, transform=val_transforms)
    # val_ds = CacheDataset(
    #     data=val_files, transform=val_transforms)
    val_org_loader = DataLoader(val_ds, batch_size=30, shuffle=False, num_workers=0)

    check_ds = Dataset(data=val_files, transform=val_transforms)
    check_loader = DataLoader(check_ds, batch_size=1)
    check_data = first(check_loader)
    image = (check_data["image"])[0].T
    print(image.mean(),image.max(),image.min())
    label = check_data["label"]
    print(f"image shape: {image.shape}, label shape: {label.shape}")
    # plot the slice [:, :, 80]
    plt.figure("check", (12, 6))
    plt.title("image")
    plt.imshow((image* 255).numpy().astype(np.uint8))
    plt.savefig('data_loader.png')
    return train_loader, val_org_loader


if __name__ == '__main__':
    config = SimpleNamespace(patchshape=(96, 96, 96))
    get_class_data_loaders(config,3)
