import torch
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from tools import shuffle_data
from preprocessing.Datasets import normalize_dataset

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Base augmentations operators."""

# ImageNet code should change this value
IMAGE_SIZE = 224


def int_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
  return int(level * maxval / 10)


def float_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval.

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
  return float(level) * maxval / 10.


def sample_level(n):
  return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
  return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
  return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
  level = int_parameter(sample_level(level), 4)
  return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
  degrees = int_parameter(sample_level(level), 30)
  if np.random.uniform() > 0.5:
    degrees = -degrees
  return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
  level = int_parameter(sample_level(level), 256)
  return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, level, 0, 0, 1, 0),
                           resample=Image.BILINEAR)


def shear_y(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, 0, level, 1, 0),
                           resample=Image.BILINEAR)


def translate_x(pil_img, level):
  level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, level, 0, 1, 0),
                           resample=Image.BILINEAR)


def translate_y(pil_img, level):
  level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, 0, 0, 1, level),
                           resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)

augmentations_augmix = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y
]

augmentations_augmix_all = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y, color, contrast, brightness, sharpness
]

class AugMixGenerator:
    def __init__(self,
                 dataset,
                 batch_size,
                 stage,
                 aug_prob_coeff= 1.,
                 mixture_width = 3,
                 mixture_depth = -1,
                 aug_severity = 1,
                 img_mean_mode=None,
                 seed=13,
                 orig_plus_aug=True):
        """
        :param dataset: (tuple) x, y, segmentation mask (optional)
        :param batch_size: (int) # of inputs in a mini-batch
        :param stage: (str) train | test
        :param aug_prob_coeff: (float) alpha in the paper
        :param aug_severity: (int) from the reposityory
        :param mixture_width: (int) from the paper
        :param mixture_depth: (int) from the paper
        :param img_mean_mode: (str) use this for image normalization
        :param seed: (int) seed for input shuffle
        :param orig_plus_aug: (bool) if True, original images will be kept in the batch along with corrupted ones
        """

        if stage not in ['train', 'test']:
            assert ValueError('invalid stage!')

        # Settings
        self.batch_size = batch_size
        self.stage = stage
        self.aug_prob_coeff = aug_prob_coeff
        self.mixture_width = mixture_width
        self.mixture_depth = mixture_depth
        self.aug_severity = aug_severity
        self.img_mean_mode = img_mean_mode
        self.seed = seed
        self.orig_plus_aug = orig_plus_aug

        # Preparation
        self.configuration()
        self.load_data(dataset)

    def configuration(self):
        self.shuffle_count = 1
        self.current_index = 0

    def shuffle(self):
        self.image_count = len(self.labels)
        self.current_index = 0
        self.images, self.labels, self.teacher_logits, _ = shuffle_data(samples=self.images,
                                                                        labels=self.labels,
                                                                        teacher_logits=self.teacher_logits,
                                                                        seed=self.seed + self.shuffle_count)
        self.shuffle_count += 1

    def load_data(self, dataset):
        self.images = dataset["images"]
        self.labels = dataset["labels"]
        self.teacher_logits = dataset["teacher_logits"] if "teacher_logits" in dataset else None

        self.len_images = len(self.images)
        self.len_labels = len(self.labels)
        assert self.len_images == self.len_labels
        self.image_count = self.len_labels

        if self.stage == 'train':
            self.images, self.labels, self.teacher_logits, _ = shuffle_data(samples=self.images,
                                                                            labels=self.labels,
                                                                            teacher_logits=self.teacher_logits,
                                                                            seed=self.seed)

    def get_batch_count(self):
        return (self.len_labels // self.batch_size) + 1

    def augmix(self, x):
        aug_list = augmentations_augmix_all

        ws = np.float32(np.random.dirichlet([self.aug_prob_coeff] * self.mixture_width))
        m = np.float32(np.random.beta(self.aug_prob_coeff, self.aug_prob_coeff))

        mix = np.zeros_like(np.array(x)).astype("float32")
        for i in range(self.mixture_width):
            x_ = x.copy()
            depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(1, 4)
            for _ in range(depth):
                op = np.random.choice(aug_list)
                x_ = op(x_, self.aug_severity)

            # Preprocessing commutes since all coefficients are convex
            mix += ws[i] * np.array(x_).astype("float32")

        mix = Image.fromarray(mix.astype("uint8"))
        x_ = Image.blend(x, mix, m)

        return x_

    def augment(self, x):

        x_ = [self.augmix(x) for _ in range(2)]

        return x_

    def get_batch(self, epoch=None):
        tensor_shape = (self.batch_size, self.images.shape[3], self.images.shape[1], self.images.shape[2])
        teacher_logits = None if self.teacher_logits is None else []
        expansion_coeff = 2

        if self.orig_plus_aug:
            labels = np.zeros(tuple([self.batch_size] + list(self.labels.shape)[1:]))
            images = torch.zeros(tensor_shape, dtype=torch.float32)
            augmented_images = [torch.zeros(tensor_shape, dtype=torch.float32) for _ in range(expansion_coeff)]
            for i in range(self.batch_size):
                # Avoid over flow
                if self.current_index > self.image_count - 1:
                    if self.stage == "train":
                        self.shuffle()
                    else:
                        self.current_index = 0

                x = Image.fromarray(self.images[self.current_index])
                y = self.labels[self.current_index]
                images[i] = normalize_dataset(x, self.img_mean_mode)
                labels[i] = y

                augmented_x = self.augment(x)
                for j in range(expansion_coeff):
                    augmented_images[j][i] = normalize_dataset(augmented_x[j], img_mean_mode=self.img_mean_mode)

                if teacher_logits is not None:
                    teacher_logits.append(self.teacher_logits[self.current_index])

                self.current_index += 1

            # Include teacher logits as soft labels if applicable
            if teacher_logits is not None:
                labels = [labels, np.array(teacher_logits)]

            batches = [(images, labels)]
            for i in range(expansion_coeff):
                batches.append((augmented_images[i], labels))

        else:
            labels = np.zeros(tuple([self.batch_size] + list(self.labels.shape)[1:]))
            augmented_images = [torch.zeros(tensor_shape, dtype=torch.float32) for _ in range(expansion_coeff)]
            for i in range(self.batch_size):
                # Avoid over flow
                if self.current_index > self.image_count - 1:
                    if self.stage == "train":
                        self.shuffle()
                    else:
                        self.current_index = 0

                x = Image.fromarray(self.images[self.current_index])
                y = self.labels[self.current_index]
                labels[i] = y

                augmented_x = self.augment(x)
                for j in range(expansion_coeff):
                    augmented_images[j][i] = normalize_dataset(augmented_x[j], img_mean_mode=self.img_mean_mode)

                if teacher_logits is not None:
                    teacher_logits.append(self.teacher_logits[self.current_index])

                self.current_index += 1

            # Include teacher logits as soft labels if applicable
            if teacher_logits is not None:
                labels = [labels, np.array(teacher_logits)]

            batches = []
            for i in range(expansion_coeff):
                batches.append((augmented_images[i], labels))

        return batches
