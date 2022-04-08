import torch
import numpy as np
from PIL import Image
from tools import shuffle_data
from preprocessing.Datasets import normalize_dataset

class ImageGenerator:
    def __init__(self,
                 dataset,
                 batch_size,
                 stage,
                 img_mean_mode=None,
                 seed=13):
        """
        :param dataset: (tuple) x, y
        :param batch_size: (int) # of inputs in a mini-batch
        :param stage: (str) train | test
        :param img_mean_mode: (str) use this for image normalization
        :param seed: (int) seed for input shuffle
        """

        if stage not in ['train', 'test']:
            assert ValueError('invalid stage!')

        # Settings
        self.batch_size = batch_size
        self.stage = stage
        self.img_mean_mode = img_mean_mode
        self.seed = seed

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
        self.images = dataset["images"] if type(dataset) is dict else dataset[0]
        self.labels = dataset["labels"] if type(dataset) is dict else dataset[1]
        self.teacher_logits = dataset["teacher_logits"] if type(dataset) is dict and "teacher_logits" in dataset else None

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

    def get_batch(self, epoch=None):
        tensor_shape = (self.batch_size, self.images.shape[3], self.images.shape[1], self.images.shape[2])
        teacher_logits = None if self.teacher_logits is None else []
        labels = np.zeros(tuple([self.batch_size] + list(self.labels.shape)[1:]))
        images = torch.zeros(tensor_shape, dtype=torch.float32)
        for i in range(self.batch_size):
            # Avoid over flow
            if self.current_index > self.image_count - 1:
                if self.stage == "train":
                    self.shuffle()
                else:
                    self.current_index = 0

            image = Image.fromarray(self.images[self.current_index])
            images[i] = normalize_dataset(image, img_mean_mode=self.img_mean_mode)
            labels[i] = self.labels[self.current_index]

            if teacher_logits is not None:
                teacher_logits.append(self.teacher_logits[self.current_index])

            self.current_index += 1

        # Include teacher logits as soft labels if applicable
        if teacher_logits is not None:
            labels = [labels, np.array(teacher_logits)]

        return [(images, labels)]
