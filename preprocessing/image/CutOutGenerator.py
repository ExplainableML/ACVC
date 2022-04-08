import torch
import numpy as np
from PIL import Image
from tools import shuffle_data
from preprocessing.Datasets import normalize_dataset

class CutOutGenerator:
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
        self.img_mean_mode=img_mean_mode
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

    def get_random_eraser(self, p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1 / 0.3, v_l=0.0, v_h=1.0):
        """
        This CutOut implementation is taken from:
            - https://github.com/yu4u/cutout-random-erasing

        ...and modified for info loss experiments

        # Arguments:
            :param p: (float) the probability that random erasing is performed
            :param s_l: (float) minimum proportion of erased area against input image
            :param s_h: (float) maximum proportion of erased area against input image
            :param r_1: (float) minimum aspect ratio of erased area
            :param r_2: (float) maximum aspect ratio of erased area
            :param v_l: (float) minimum value for erased area
            :param v_h: (float) maximum value for erased area
            :param fill: (str) fill-in mode for the cropped area

        :return: (np.array) augmented image
        """
        def eraser(orig_img):
            input_img = np.copy(orig_img)
            if input_img.ndim == 3:
                img_h, img_w, img_c = input_img.shape
            elif input_img.ndim == 2:
                img_h, img_w = input_img.shape

            p_1 = np.random.rand()

            if p_1 > p:
                return input_img

            while True:
                s = np.random.uniform(s_l, s_h) * img_h * img_w
                r = np.random.uniform(r_1, r_2)
                w = int(np.sqrt(s / r))
                h = int(np.sqrt(s * r))
                left = np.random.randint(0, img_w)
                top = np.random.randint(0, img_h)

                if left + w <= img_w and top + h <= img_h:
                    break

            input_img[top:top + h, left:left + w] = 0

            return input_img

        return eraser

    def cutout(self, x):

        eraser = self.get_random_eraser()
        x_ = eraser(x)

        return x_

    def get_batch(self, epoch=None):
        tensor_shape = (self.batch_size, self.images.shape[3], self.images.shape[1], self.images.shape[2])

        if self.orig_plus_aug:
            labels = np.zeros(tuple([self.batch_size] + list(self.labels.shape)[1:]))
            images = torch.zeros(tensor_shape, dtype=torch.float32)
            augmented_images = torch.zeros(tensor_shape, dtype=torch.float32)
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

                augmented_x = self.cutout(x)
                augmented_images[i] = normalize_dataset(Image.fromarray(augmented_x), img_mean_mode=self.img_mean_mode)

                self.current_index += 1

            batches = [(images, labels), (augmented_images, labels)]

        else:
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

                augmented_x = self.cutout(x)
                images[i] = normalize_dataset(Image.fromarray(augmented_x), img_mean_mode=self.img_mean_mode)
                labels[i] = y

                self.current_index += 1

            batches = [(images, labels)]

        return batches
