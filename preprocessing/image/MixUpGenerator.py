import copy
import torch
import numpy as np
from PIL import Image
from tools import shuffle_data
from preprocessing.Datasets import normalize_dataset

class MixUpGenerator:
    def __init__(self,
                 dataset,
                 batch_size,
                 stage,
                 img_mean_mode=None,
                 seed=13,
                 orig_plus_aug=True):
        """
        :param dataset: (tuple) x, y, segmentation mask (optional)
        :param batch_size: (int) # of inputs in a mini-batch
        :param stage: (str) train | test
        :param img_mean_mode: (str) use this for image normalization
        :param seed: (int) seed for input shuffle
        :param orig_plus_aug: (bool) if True, original images will be kept in the batch along with corrupted ones
        """

        if stage not in ['train', 'test']:
            assert ValueError('invalid stage!')

        # Settings
        self.batch_size = batch_size
        self.stage = stage
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

    def mixup(self, image_batch, label_batch, beta=1.0):
        l_param = np.random.beta(beta, beta, self.batch_size)
        rand_index = np.random.permutation(self.batch_size)

        x = image_batch.detach().clone()
        y = np.zeros_like(label_batch)

        for i in range(self.batch_size):
            x[i] = l_param[i] * x[i] + (1 - l_param[i]) * x[rand_index[i]]
            y[i] = label_batch[rand_index[i]]

        return x, (label_batch, y, l_param)

    def get_batch(self, epoch=None):
        tensor_shape = (self.batch_size, self.images.shape[3], self.images.shape[1], self.images.shape[2])
        labels = np.zeros(tuple([self.batch_size] + list(self.labels.shape)[1:]))
        images = torch.zeros(tensor_shape, dtype=torch.float32)
        for i in range(self.batch_size):
            # Avoid over flow
            if self.current_index > self.image_count - 1:
                if self.stage == "train":
                    self.shuffle()
                else:
                    self.current_index = 0

            x = self.images[self.current_index]
            y = self.labels[self.current_index]
            images[i] = normalize_dataset(Image.fromarray(x), img_mean_mode=self.img_mean_mode)
            labels[i] = y

            self.current_index += 1

        augmented_images, augmented_labels = self.mixup(images, labels)

        if self.orig_plus_aug:
            batches = [(images, (labels, labels, (np.ones_like(labels)))), (augmented_images, augmented_labels)]

        else:
            batches = [(augmented_images, augmented_labels)]

        return batches
