import torch
import numpy as np
from tools import shuffle_data
from preprocessing.Datasets import normalize_dataset
from PIL import Image as PILImage
from scipy.stats import truncnorm
from preprocessing.image.RandAugmentGenerator import RandAugment
from imagecorruptions import corrupt, get_corruption_names

class ACVCGenerator:
    def __init__(self,
                 dataset,
                 batch_size,
                 stage,
                 epochs,
                 corruption_mode,
                 corruption_dist,
                 img_mean_mode=None,
                 rand_aug=False,
                 seed=13,
                 orig_plus_aug=True):
        """
        19 out of 22 corruptions used in this generator are taken from ImageNet-10.C:
            - https://github.com/bethgelab/imagecorruptions

        :param dataset: (tuple) x, y, segmentation mask (optional)
        :param batch_size: (int) # of inputs in a mini-batch
        :param stage: (str) train | test
        :param epochs: (int) # of full training passes
        :param corruption_mode: (str) requied for VisCo experiments
        :param corruption_dist: (str) requied to determine the corruption rate
        :param img_mean_mode: (str) use this to revert the image into its original form before augmentation
        :param rand_aug: (bool) enable/disable on-the-fly random data augmentation
        :param seed: (int) seed for input shuffle
        :param orig_plus_aug: (bool) if True, original images will be kept in the batch along with corrupted ones
        """

        if stage not in ['train', 'test']:
            assert ValueError('invalid stage!')

        # Settings
        self.batch_size = batch_size
        self.stage = stage
        self.epochs = epochs
        self.corruption_mode = corruption_mode
        self.corruption_dist = corruption_dist
        self.img_mean_mode = img_mean_mode
        self.rand_aug = rand_aug
        self.seed = seed
        self.orig_plus_aug = orig_plus_aug

        # Preparation
        self.configuration()
        self.load_data(dataset)
        self.random_augmentation = RandAugment(1, 5)
        if self.img_mean_mode is not None:
            self.img_mean = np.load("img_mean.npy")

    def configuration(self):
        self.shuffle_count = 1
        self.current_index = 0

    def shuffle(self):
        self.image_count = len(self.labels)
        self.current_index = 0
        self.images, self.labels, self.teacher_logits, self.segmentation_masks = shuffle_data(samples=self.images,
                                                                                              labels=self.labels,
                                                                                              teacher_logits=self.teacher_logits,
                                                                                              segmentation_masks=self.segmentation_masks,
                                                                                              seed=self.seed + self.shuffle_count)
        self.shuffle_count += 1

    def load_data(self, dataset):
        self.images = dataset["images"]
        self.labels = dataset["labels"]
        self.teacher_logits = dataset["teacher_logits"] if "teacher_logits" in dataset else None
        self.segmentation_masks = dataset["segmentation_masks"] if "segmentation_masks" in dataset else None

        self.len_images = len(self.images)
        self.len_labels = len(self.labels)
        assert self.len_images == self.len_labels
        self.image_count = self.len_labels

        if self.stage == 'train':
            self.images, self.labels, self.teacher_logits, self.segmentation_masks = shuffle_data(samples=self.images,
                                                                                                  labels=self.labels,
                                                                                                  teacher_logits=self.teacher_logits,
                                                                                                  segmentation_masks=self.segmentation_masks,
                                                                                                  seed=self.seed)

    def get_batch_count(self):
        return (self.len_labels // self.batch_size) + 1

    def get_truncated_normal(self, mean=0, sd=1, low=0, upp=10):
        return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

    def get_severity(self):
        return np.random.randint(1, 6)

    def draw_cicle(self, shape, diamiter):
        """
        Input:
        shape    : tuple (height, width)
        diameter : scalar

        Output:
        np.array of shape  that says True within a circle with diamiter =  around center
        """
        assert len(shape) == 2
        TF = np.zeros(shape, dtype="bool")
        center = np.array(TF.shape) / 2.0

        for iy in range(shape[0]):
            for ix in range(shape[1]):
                TF[iy, ix] = (iy - center[0]) ** 2 + (ix - center[1]) ** 2 < diamiter ** 2
        return TF

    def filter_circle(self, TFcircle, fft_img_channel):
        temp = np.zeros(fft_img_channel.shape[:2], dtype=complex)
        temp[TFcircle] = fft_img_channel[TFcircle]
        return temp

    def inv_FFT_all_channel(self, fft_img):
        img_reco = []
        for ichannel in range(fft_img.shape[2]):
            img_reco.append(np.fft.ifft2(np.fft.ifftshift(fft_img[:, :, ichannel])))
        img_reco = np.array(img_reco)
        img_reco = np.transpose(img_reco, (1, 2, 0))
        return img_reco

    def high_pass_filter(self, x, severity):
        x = x.astype("float32") / 255.
        c = [.01, .02, .03, .04, .05][severity - 1]

        d = int(c * x.shape[0])
        TFcircle = self.draw_cicle(shape=x.shape[:2], diamiter=d)
        TFcircle = ~TFcircle

        fft_img = np.zeros_like(x, dtype=complex)
        for ichannel in range(fft_img.shape[2]):
            fft_img[:, :, ichannel] = np.fft.fftshift(np.fft.fft2(x[:, :, ichannel]))

        # For each channel, pass filter
        fft_img_filtered = []
        for ichannel in range(fft_img.shape[2]):
            fft_img_channel = fft_img[:, :, ichannel]
            temp = self.filter_circle(TFcircle, fft_img_channel)
            fft_img_filtered.append(temp)
        fft_img_filtered = np.array(fft_img_filtered)
        fft_img_filtered = np.transpose(fft_img_filtered, (1, 2, 0))
        x = np.clip(np.abs(self.inv_FFT_all_channel(fft_img_filtered)), a_min=0, a_max=1)

        x = PILImage.fromarray((x * 255.).astype("uint8"))
        return x

    def constant_amplitude(self, x, severity):
        """
        A visual corruption based on amplitude information of a Fourier-transformed image

        Adopted from: https://github.com/MediaBrain-SJTU/FACT
        """
        x = x.astype("float32") / 255.
        c = [.05, .1, .15, .2, .25][severity - 1]

        # FFT
        x_fft = np.fft.fft2(x, axes=(0, 1))
        x_abs, x_pha = np.fft.fftshift(np.abs(x_fft), axes=(0, 1)), np.angle(x_fft)

        # Amplitude replacement
        beta = 1.0 - c
        x_abs = np.ones_like(x_abs) * max(0, beta)

        # Inverse FFT
        x_abs = np.fft.ifftshift(x_abs, axes=(0, 1))
        x = x_abs * (np.e ** (1j * x_pha))
        x = np.real(np.fft.ifft2(x, axes=(0, 1)))

        x = PILImage.fromarray((x * 255.).astype("uint8"))
        return x

    def phase_scaling(self, x, severity):
        """
        A visual corruption based on phase information of a Fourier-transformed image

        Adopted from: https://github.com/MediaBrain-SJTU/FACT
        """
        x = x.astype("float32") / 255.
        c = [.1, .2, .3, .4, .5][severity - 1]

        # FFT
        x_fft = np.fft.fft2(x, axes=(0, 1))
        x_abs, x_pha = np.fft.fftshift(np.abs(x_fft), axes=(0, 1)), np.angle(x_fft)

        # Phase scaling
        alpha = 1.0 - c
        x_pha = x_pha * max(0, alpha)

        # Inverse FFT
        x_abs = np.fft.ifftshift(x_abs, axes=(0, 1))
        x = x_abs * (np.e ** (1j * x_pha))
        x = np.real(np.fft.ifft2(x, axes=(0, 1)))

        x = PILImage.fromarray((x * 255.).astype("uint8"))
        return x

    def apply_corruption(self, x, corruption_name):
        severity = self.get_severity()

        custom_corruptions = {"high_pass_filter": self.high_pass_filter,
                              "constant_amplitude": self.constant_amplitude,
                              "phase_scaling": self.phase_scaling}

        if corruption_name in get_corruption_names('all'):
            x = corrupt(x, corruption_name=corruption_name, severity=severity)
            x = PILImage.fromarray(x)

        elif corruption_name in custom_corruptions:
            x = custom_corruptions[corruption_name](x, severity=severity)

        else:
            assert True, "%s is not a supported corruption!" % corruption_name

        return x

    def acvc(self, x):
        i = np.random.randint(0, 22)
        corruption_func = {0: "fog",
                           1: "snow",
                           2: "frost",
                           3: "spatter",
                           4: "zoom_blur",
                           5: "defocus_blur",
                           6: "glass_blur",
                           7: "gaussian_blur",
                           8: "motion_blur",
                           9: "speckle_noise",
                           10: "shot_noise",
                           11: "impulse_noise",
                           12: "gaussian_noise",
                           13: "jpeg_compression",
                           14: "pixelate",
                           15: "elastic_transform",
                           16: "brightness",
                           17: "saturate",
                           18: "contrast",
                           19: "high_pass_filter",
                           20: "constant_amplitude",
                           21: "phase_scaling"}
        return self.apply_corruption(x, corruption_func[i])

    def corruption(self, x, segmentation_mask=None):
        if self.rand_aug and np.random.uniform(0, 1) > 0.5:
            x_ = self.random_augmentation(PILImage.fromarray(x))

        else:
            x_ = np.copy(x)
            x_ = self.acvc(x_)

        return x_

    def get_batch(self, epoch=None):
        tensor_shape = (self.batch_size, self.images.shape[3], self.images.shape[1], self.images.shape[2])
        teacher_logits = None if self.teacher_logits is None else []

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
                mask = None if self.segmentation_masks is None else self.segmentation_masks[self.current_index]
                images[i] = normalize_dataset(PILImage.fromarray(x), img_mean_mode=self.img_mean_mode)
                labels[i] = y

                augmented_x = self.corruption(x, mask)
                augmented_images[i] = normalize_dataset(augmented_x, img_mean_mode=self.img_mean_mode)

                if teacher_logits is not None:
                    teacher_logits.append(self.teacher_logits[self.current_index])

                self.current_index += 1

            # Include teacher logits as soft labels if applicable
            if teacher_logits is not None:
                labels = [labels, np.array(teacher_logits)]

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
                mask = None if self.segmentation_masks is None else self.segmentation_masks[self.current_index]

                augmented_x = self.corruption(x, mask)
                images[i] = normalize_dataset(augmented_x, img_mean_mode=self.img_mean_mode)
                labels[i] = y

                if teacher_logits is not None:
                    teacher_logits.append(self.teacher_logits[self.current_index])

                self.current_index += 1

            # Include teacher logits as soft labels if applicable
            if teacher_logits is not None:
                labels = [labels, np.array(teacher_logits)]

            batches = [(images, labels)]

        return batches