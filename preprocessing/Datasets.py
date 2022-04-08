from __future__ import print_function, absolute_import, division
import os
import torchvision
import numpy as np
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from tools import log, resize_image

def preprocess_dataset(x, train, img_mean_mode):
    # Compute image mean if applicable
    if img_mean_mode is not None:
        if train:

            if img_mean_mode == "per_channel":
                x_ = np.copy(x)
                x_ = x_.astype('float32') / 255.0
                img_mean = np.array([np.mean(x_[:, :, :, 0]), np.mean(x_[:, :, :, 1]), np.mean(x_[:, :, :, 2])])

            elif img_mean_mode == "imagenet":
                img_mean = np.array([0.485, 0.456, 0.406])

            else:
                raise Exception("Invalid img_mean_mode..!")
            np.save("img_mean.npy", img_mean)

    return x

def normalize_dataset(x, img_mean_mode):
    if img_mean_mode is not None:

        if img_mean_mode == "imagenet":
            img_mean = np.array([0.485, 0.456, 0.406])
            img_std = np.array([0.229, 0.224, 0.225])

        else:
            assert os.path.exists("img_mean.npy"), "Image mean file cannot be found!"
            img_mean = np.load("img_mean.npy")
            img_std = np.array([1.0, 1.0, 1.0])

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=img_mean, std=img_std)
        ])

    else:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])

    x_ = transform(x)

    return x_

def load_PACS(subset="photo", train=True, img_mean_mode="imagenet", distillation=False, data_dir="../../datasets"):
    subset = "art_painting" if subset.lower() == "art" else subset.lower()
    root_path = os.path.join(os.path.dirname(__file__), data_dir, 'PACS')
    data_path = os.path.join(root_path, subset.lower())
    classes = {"dog": 0, "elephant": 1, "giraffe": 2, "guitar": 3, "horse": 4, "house": 5, "person": 6}
    img_dim = (224, 224)

    imagedata = []
    labels = []
    teacher_logits = None
    if subset == "photo":
        label_file = "photo_train.txt" if train else "photo_val.txt"
        label_file = os.path.join(root_path, label_file)

        # Gather images and labels
        with open(label_file, "r") as f_label:
            for line in f_label:
                temp = line[:-1].split(" ")
                img = Image.open(os.path.join(root_path, temp[0]))
                img = resize_image(img, img_dim)
                imagedata.append(np.array(img))
                labels.append(int(temp[1]) - 1)

        imagedata = np.array(imagedata)
        labels = np.array(labels)

        # Include teacher logits as well if applicable
        if train and distillation:
            logit_file = os.path.join(root_path, "teacher_logits.npy")
            assert os.path.exists(logit_file), "Teacher logits cannot be found for PACS!"
            teacher_logits = np.load(logit_file)

    else:
        for class_dir in os.listdir(data_path):
            label = classes[class_dir]
            path = os.path.join(data_path, class_dir)

            for img_file in os.listdir(path):
                if img_file.endswith("jpg") or img_file.endswith("png"):
                    img_path = os.path.join(path, img_file)
                    img = Image.open(img_path)
                    img = resize_image(img, img_dim)
                    imagedata.append(np.array(img))
                    labels.append(label)

        imagedata = np.array(imagedata)
        labels = np.array(labels)

    # Normalize the data
    imagedata = preprocess_dataset(imagedata, train=train, img_mean_mode=img_mean_mode)
    result = {"images": imagedata, "labels": labels} if train else (imagedata, labels)

    if not teacher_logits is None:
        result["teacher_logits"] = teacher_logits

    return result

def prep_COCO(save_masks_as_image=True, data_dir="../../datasets"):
    data_path = os.path.join(os.path.dirname(__file__), data_dir, 'COCO')
    classes = {"airplane": 0, "bicycle": 1, "bus": 2, "car": 3, "horse": 4, "knife": 5, "motorcycle": 6,
               "skateboard": 7, "train": 8, "truck": 9}
    object_scene_ratio_lower_threshold = 0.1
    object_scene_ratio_upper_threshold = 1.0
    class_names = [""] * 10
    class_ids = [0] * 10
    img_dim = (224, 224)

    # Prepare training data
    train_data_path = os.path.join(data_path, "annotations", "instances_train2017.json")
    coco = COCO(train_data_path)

    catIds = coco.getCatIds()
    cats = coco.loadCats(catIds)
    for cat in cats:
        cat_name = cat["name"]
        if cat_name in classes:
            class_names[classes[cat_name]] = cat_name
            class_ids[classes[cat_name]] = cat["id"]

    total_img_count = 0
    landing_dir = os.path.join(data_path, "downloads", "train2017")
    target_dir = os.path.join(data_path, "train2017")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

        for i in range(len(class_ids)):
            class_id = class_ids[i]
            class_name = class_names[i]
            target_class_dir = os.path.join(target_dir, class_name)

            if not os.path.exists(target_class_dir):
                os.makedirs(target_class_dir)

                img_ids = coco.getImgIds(catIds=class_id)
                for img_id in tqdm(img_ids):
                    img_info = coco.loadImgs(img_id)
                    assert len(img_info) == 1, "Image retrieval problem in COCO training set!"
                    img_info = img_info[0]

                    ann_id = coco.getAnnIds(imgIds=img_id, catIds=class_id)
                    anns = coco.loadAnns(ann_id)

                    # Generate binary mask
                    mask = np.zeros((img_info['height'], img_info['width']))
                    for j in range(len(anns)):
                        mask = np.maximum(coco.annToMask(anns[j]), mask)

                    if object_scene_ratio_lower_threshold < (
                            np.sum(mask) / mask.size) <= object_scene_ratio_upper_threshold:
                        total_img_count += 1

                        # Copy relevant image to dest and save its corresponding binary mask
                        source_path = os.path.join(landing_dir, img_info["file_name"])
                        assert os.path.exists(source_path), "Image is not found in the source path!"
                        dest_path = os.path.join(target_class_dir, img_info["file_name"])
                        img = Image.open(source_path)
                        img = resize_image(img, img_dim)
                        img.save(dest_path)

                        if save_masks_as_image:
                            mask_img_path = os.path.join(target_class_dir, "%s_mask.jpg" % img_info["file_name"].split(".jpg")[0])
                            mask_img = np.array(mask * 255, dtype=np.uint8)
                            mask_img = Image.fromarray(mask_img)
                            mask_img = resize_image(mask_img, img_dim)
                            mask_img.save(mask_img_path)
                        else:
                            mask_path = os.path.join(target_class_dir, img_info["file_name"].replace("jpg", "npy"))
                            np.save(mask_path, mask)
    log("%s COCO training images are prepared." % total_img_count)

    # Prepare validation data
    val_data_path = os.path.join(data_path, "annotations", "instances_val2017.json")
    coco = COCO(val_data_path)

    total_img_count = 0
    landing_dir = os.path.join(data_path, "downloads", "val2017")
    target_dir = os.path.join(data_path, "val2017")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

        for i in range(len(class_ids)):
            class_id = class_ids[i]
            class_name = class_names[i]
            target_class_dir = os.path.join(target_dir, class_name)

            if not os.path.exists(target_class_dir):
                os.makedirs(target_class_dir)

                img_ids = coco.getImgIds(catIds=class_id)
                for img_id in tqdm(img_ids):
                    img_info = coco.loadImgs(img_id)
                    assert len(img_info) == 1, "Image retrieval problem in COCO validation set!"
                    img_info = img_info[0]

                    ann_id = coco.getAnnIds(imgIds=img_id, catIds=class_id)
                    anns = coco.loadAnns(ann_id)

                    # Generate binary mask
                    mask = np.zeros((img_info['height'], img_info['width']))
                    for j in range(len(anns)):
                        mask = np.maximum(coco.annToMask(anns[j]), mask)

                    if object_scene_ratio_lower_threshold < (np.sum(mask) / mask.size) <= object_scene_ratio_upper_threshold:
                        total_img_count += 1

                        # Copy relevant image to dest and save its corresponding binary mask
                        source_path = os.path.join(landing_dir, img_info["file_name"])
                        assert os.path.exists(source_path), "Image is not found in the source path!"
                        dest_path = os.path.join(target_class_dir, img_info["file_name"])
                        img = Image.open(source_path)
                        img = resize_image(img, img_dim)
                        img.save(dest_path)

                        if save_masks_as_image:
                            mask_img_path = os.path.join(target_class_dir, "%s_mask.jpg" % img_info["file_name"].split(".jpg")[0])
                            mask_img = np.array(mask * 255, dtype=np.uint8)
                            mask_img = Image.fromarray(mask_img)
                            mask_img = resize_image(mask_img, img_dim)
                            mask_img.save(mask_img_path)
                        else:
                            mask_path = os.path.join(target_class_dir, img_info["file_name"].replace("jpg", "npy"))
                            np.save(mask_path, mask)

    log("%s COCO validation images are prepared." % total_img_count)

def load_COCO(train=True, first_run=False, img_mean_mode="imagenet", distillation=False, data_dir="../../datasets"):
    if first_run:
        prep_COCO(data_dir=data_dir)

    data_path = os.path.join(os.path.dirname(__file__), data_dir, 'COCO')
    classes = {"airplane": 0, "bicycle": 1, "bus": 2, "car": 3, "horse": 4, "knife": 5, "motorcycle": 6,
               "skateboard": 7, "train": 8, "truck": 9}
    per_class_img_limit = 1000
    img_dim = (224, 224)

    if train:
        # Training data
        x_train = []
        y_train = []
        masks = []
        for class_dir in classes:
            label = classes[class_dir]
            path = os.path.join(data_path, "train2017", class_dir)

            file_list = [i for i in sorted(os.listdir(path)) if i.endswith("jpg") and "mask" not in i][:per_class_img_limit]
            for img_file in file_list:
                img_path = os.path.join(path, img_file)
                img = Image.open(img_path)
                img = resize_image(img, img_dim)
                mask_path = os.path.join(path, "%s_mask.jpg" % img_file.split(".jpg")[0])
                mask = Image.open(mask_path).convert('L')
                mask = resize_image(mask, img_dim)
                x_train.append(np.array(img))
                y_train.append(label)
                masks.append(np.array(mask))

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        masks = np.array(masks)

        # Normalize the data
        x_train = preprocess_dataset(x_train, train=True, img_mean_mode=img_mean_mode)
        masks = masks.astype('float32') / 255.0
        masks = masks[:, :, :, np.newaxis]
        result = {"images": x_train, "labels": y_train, "segmentation_masks": masks}

        if distillation:
            logit_file = os.path.join(data_path, "teacher_logits.npy")
            assert os.path.exists(logit_file), "Teacher logits cannot be found for COCO!"
            result["teacher_logits"] = np.load(logit_file)

    else:
        # Validation data
        x_val = []
        y_val = []
        for class_dir in classes:
            label = classes[class_dir]
            path = os.path.join(data_path, "val2017", class_dir)

            file_list = [i for i in sorted(os.listdir(path)) if i.endswith("jpg") and "mask" not in i]
            for img_file in file_list:
                img_path = os.path.join(path, img_file)
                img = Image.open(img_path)
                img = resize_image(img, img_dim)
                x_val.append(np.array(img))
                y_val.append(label)

        x_val = np.array(x_val)
        y_val = np.array(y_val)

        # Normalize the data
        x_val = preprocess_dataset(x_val, train=False, img_mean_mode=img_mean_mode)
        result = (x_val, y_val)

    return result

def load_DomainNet(subset, img_mean_mode="imagenet", data_dir="../../datasets"):
    data_path = os.path.join(os.path.dirname(__file__), data_dir,  'DomainNet', subset.lower())
    classes = {"airplane": 0, "bicycle": 1, "bus": 2, "car": 3, "horse": 4, "knife": 5, "motorcycle": 6,
               "skateboard": 7, "train": 8, "truck": 9}
    img_dim = (224, 224)

    imagedata = []
    labels = []
    for class_dir in classes:
        label = classes[class_dir]
        path = os.path.join(data_path, class_dir)

        for img_file in os.listdir(path):
            if img_file.endswith("jpg") or img_file.endswith("png"):
                img_path = os.path.join(path, img_file)
                img = Image.open(img_path)
                img = resize_image(img, img_dim,)
                imagedata.append(np.array(img))
                labels.append(label)

    imagedata = np.array(imagedata)
    imagedata = preprocess_dataset(imagedata, train=False, img_mean_mode=img_mean_mode)
    labels = np.array(labels)

    return imagedata, labels

def load_FullDomainNet(subset, train=True, img_mean_mode="imagenet", distillation=False, data_dir="../../datasets"):
    data_path = os.path.join(os.path.dirname(__file__), data_dir,  'FullDomainNet')
    img_dim = (224, 224)
    subset = subset.lower()
    if subset == "real":
        labelfile = os.path.join(data_path, "real_train.txt") if train else os.path.join(data_path, "real_test.txt")
    else:
        labelfile = os.path.join(data_path, "fast.txt") #os.path.join(data_path, "%s.txt" % subset)

    # Gather image paths and labels
    imagepath = []
    labels = []
    with open(labelfile, "r") as f_label:
        for line in f_label:
            temp = line[:-1].split(" ")
            imagepath.append(temp[0])
            labels.append(int(temp[1]))
    labels = np.array(labels)

    imagedata = np.empty([labels.shape[0]] + list(img_dim) + [3], dtype="uint8")
    for i in range(len(labels)):
        img_path = os.path.join(data_path, imagepath[i])
        img = Image.open(img_path)
        img = resize_image(img, img_dim)
        imagedata[i] = np.array(img)

    imagedata = preprocess_dataset(imagedata, train=train, img_mean_mode=img_mean_mode)

    return {"images": imagedata, "labels": labels} if train else (imagedata, labels)