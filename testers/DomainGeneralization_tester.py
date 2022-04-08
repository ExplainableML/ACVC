import os
import json
import torch
import numpy as np
from PIL import Image
from tools import log, compute_accuracy
from losses.AttentionConsistency import AttentionConsistency
from losses.Distillation import Distillation
from losses.JSDivergence import JSDivergence
from preprocessing import Datasets
from preprocessing.image.ImageGenerator import ImageGenerator
from preprocessing.image.ACVCGenerator import ACVCGenerator
from preprocessing.image.MixUpGenerator import MixUpGenerator
from preprocessing.image.CutOutGenerator import CutOutGenerator
from preprocessing.image.CutMixGenerator import CutMixGenerator
from preprocessing.image.AblationGenerator import AblationGenerator
from preprocessing.image.RandAugmentGenerator import RandAugmentGenerator
from preprocessing.image.AugMixGenerator import AugMixGenerator

class LR_Scheduler:
    def __init__(self, base_lr, dataset, optimizer):
        self.base_lr = base_lr
        self.dataset = dataset
        self.optimizer = optimizer
        self.epoch = 0

    def step(self):
        self.epoch += 1
        lr = self.base_lr

        if self.epoch > 24:
            lr *= 1e-1

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

class DomainGeneralization_tester:
    def __init__(self, train_dataset, test_dataset, img_mean_mode=None, distillation=False, wait=False, data_dir=None):
        """
        Conditional constructor to enable creating testers without immediately triggering the dataset loader.

        # Arguments
            :param train_dataset: (str) name of the training set
            :param test_dataset: (str) name of the test set
            :param img_mean_mode: (str) image mean subtraction mode for dataset preprocessing
            :param distillation: (bool) enable/disable knowledge distillation loss
            :param wait: (bool) If True, then the constructor will onyl create the instance and
                                wait for manual activation to actually load the dataset
            :param data_dir: (str) relative filepath to datasets
        """
        # Base class variables
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        self.dataset = None
        self.model = None
        self.x_test = None
        self.y_test = None
        self.training_set_size = None
        self.img_mean_mode = img_mean_mode
        self.distillation = distillation
        self.data_dir = os.path.join(ROOT_DIR, "../../datasets") if data_dir is None else data_dir

        # Loss func init
        self.classification_loss = None
        self.contrastive_loss = None
        self.distillation_loss = None

        # Extra config
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        if not wait:
            self.activate()

    def activate(self):
        """
        Call this function to manually start dataset deployment
        """

        # Deploy the dataset
        if "COCO" == self.train_dataset:
            self.dataset = Datasets.load_COCO(first_run=False, train=True, img_mean_mode=self.img_mean_mode, distillation=self.distillation, data_dir=self.data_dir)
            self.x_test, self.y_test = Datasets.load_COCO(first_run=False, train=False, img_mean_mode=self.img_mean_mode, data_dir=self.data_dir)

        elif "FullDomainNet" in self.train_dataset:
            subset = self.train_dataset.split(":")[1]
            self.dataset = Datasets.load_FullDomainNet(subset, train=True, img_mean_mode=self.img_mean_mode, distillation=self.distillation, data_dir=self.data_dir)
            self.x_test, self.y_test = Datasets.load_FullDomainNet(subset, train=False, img_mean_mode=self.img_mean_mode, data_dir=self.data_dir)

        elif "PACS" in self.train_dataset:
            subset = self.train_dataset.split(":")[1]
            self.dataset = Datasets.load_PACS(subset, train=True, img_mean_mode=self.img_mean_mode, distillation=self.distillation, data_dir=self.data_dir)
            self.x_test, self.y_test = Datasets.load_PACS(subset, train=False, img_mean_mode=self.img_mean_mode, data_dir=self.data_dir)

        else:
            assert False, "Train dataset: %s is not supported yet!" % self.train_dataset

        # Support for multiple test sets
        self.x_test = {self.train_dataset: self.x_test}
        self.y_test = {self.train_dataset: self.y_test}

        for test_set in self.test_dataset:

            if "FullDomainNet" in self.train_dataset:
                subset = test_set.split(":")[1]
                x_temp, y_temp = Datasets.load_FullDomainNet(subset, train=False, img_mean_mode=self.img_mean_mode, data_dir=self.data_dir)
                self.x_test[test_set] = x_temp
                self.y_test[test_set] = y_temp

            elif "DomainNet" in test_set:
                subset = test_set.split(":")[1]
                x_temp, y_temp = Datasets.load_DomainNet(subset, img_mean_mode=self.img_mean_mode, data_dir=self.data_dir)
                self.x_test[test_set] = x_temp
                self.y_test[test_set] = y_temp

            elif "PACS" in test_set:
                subset = test_set.split(":")[1]
                x_temp, y_temp = Datasets.load_PACS(subset, train=False, img_mean_mode=self.img_mean_mode, data_dir=self.data_dir)
                self.x_test[test_set] = x_temp
                self.y_test[test_set] = y_temp

            else:
                assert False, "Test dataset: %s is not supported yet!" % test_set

    def record_generalization_results(self, result_dict, path="generalization.json"):
        if result_dict is not None:
            # Load the previous records if exist
            hist_cache = {}
            if os.path.isfile(path):
                with open(path, "r") as hist_file:
                    hist_cache = json.load(hist_file)

            # Record new results
            for model_name in result_dict:
                if model_name in hist_cache:
                    hist_cache[model_name].append(result_dict[model_name])
                else:
                    hist_cache[model_name] = [result_dict[model_name]]

            # Save the updated records
            with open(path, "w+") as hist_file:
                json.dump(hist_cache, hist_file)

    def get_n_classes(self):

        if self.train_dataset == "COCO":
            return 10

        elif "FullDomainNet" in self.train_dataset:
            return 345

        elif "PACS" in self.train_dataset:
            return 7

        else:
            assert False, "Error: update tester.get_n_classes() for %s dataset" % self.train_dataset

    def get_optimizer(self, optimizer, lr=None, momentum=None, weight_decay=0.):
        selected_optimizer = optimizer.lower()
        if selected_optimizer == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-7, weight_decay=weight_decay)
        elif selected_optimizer == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif selected_optimizer == "nsgd":
            return torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
        else:
            return None

    def get_contrastive_element(self, loss):
        if loss == "AttentionConsistency":
            return "CAM"

        elif loss == "JSDivergence":
            return "Predictions"

        else:
            raise ValueError("Unsupported contrastive loss")

    def get_loss_func(self, loss, temperature=None):
        if loss == "CrossEntropy":
            return torch.nn.CrossEntropyLoss()

        elif loss == "AttentionConsistency":
            return AttentionConsistency(T=temperature)

        elif loss == "Distillation":
            assert temperature is not None, "Distillation requires temperature as an argument"
            return Distillation(temperature=temperature)

        elif loss == "JSDivergence":
            return JSDivergence()

        else:
            return None

    def get_multi_class_loss(self, outputs, y1, y2, l_param):
        return torch.mean(self.classification_loss(outputs, y1) * l_param + self.classification_loss(outputs, y2) * (1. - l_param))

    def test(self, x, y):
        self.model.eval()

        with torch.no_grad():
            preds = []
            for i in range(x.shape[0]):
                img = np.copy(x[i])
                img = Image.fromarray(img)
                img = Datasets.normalize_dataset(img, img_mean_mode=self.img_mean_mode).cuda()
                pred = self.model(img[None, ...])[-1]['Predictions']
                pred = pred.cpu().data.numpy()
                preds.append(pred)

            preds = np.array(preds)
            accuracy = compute_accuracy(predictions=preds, labels=y)

        return accuracy

    def validate(self):
        self.model.eval()
        batch_count = self.valGenerator.get_batch_count()

        with torch.no_grad():
            val_acc = 0
            val_loss = 0
            for i in range(batch_count):
                batches = self.valGenerator.get_batch()

                for x_val, y_val in batches:
                    images = x_val.cuda()
                    labels = torch.from_numpy(y_val).long().cuda()
                    outputs, end_points = self.model(images)

                    predictions = end_points['Predictions']
                    predictions = predictions.cpu().data.numpy()

                    val_loss += self.classification_loss(outputs, labels).cpu().item()
                    val_acc += compute_accuracy(predictions=predictions, labels=y_val)

        # Switch back to training mode
        self.model.train()

        return val_loss / batch_count, val_acc / batch_count

    def train(self, epochs, multi_class=False):
        hist = {"acc": [], "loss": [], "val_acc": [], "val_loss": []}
        batch_count = self.trainGenerator.get_batch_count()

        for epoch in range(epochs):
            train_acc = 0
            train_loss = 0
            self.model.train()
            for i in range(batch_count):
                batches = self.trainGenerator.get_batch(epoch)
                losses = []

                end_points_list = {}
                for x_train, y_train in batches:
                    # Forward with the adapted parameters
                    inputs = x_train.cuda()
                    outputs, end_points = self.model(x=inputs)

                    # Knowledge distillation loss
                    if self.distillation_loss is not None:
                        teacher_logits = torch.from_numpy(y_train[1]).cuda().squeeze()
                        y_train = y_train[0]
                        losses.append(self.distillation_loss(outputs, teacher_logits))

                    # Accumulate orig + augmented image representation/embedding if there is a contrastive loss
                    if self.contrastive_loss is not None:
                        for loss in self.contrastive_loss:
                            contrastive_element = self.get_contrastive_element(loss.name)
                            if contrastive_element not in end_points_list:
                                end_points_list[contrastive_element] = []
                            end_points_list[contrastive_element].append(end_points[contrastive_element])

                    # Classification loss
                    if multi_class:
                        y1 = torch.from_numpy(y_train[0]).long().cuda()
                        y2 = torch.from_numpy(y_train[1]).long().cuda()
                        l_param = torch.from_numpy(y_train[2]).cuda()
                        loss = self.get_multi_class_loss(outputs, y1, y2, l_param)
                        y_train = y_train[0]

                    else:
                        labels = torch.from_numpy(y_train).long().cuda()
                        loss = self.classification_loss(outputs, labels)
                    losses.append(loss)
                    train_loss += loss.cpu().item()

                    # Acc
                    predictions = end_points['Predictions']
                    predictions = predictions.cpu().data.numpy()
                    train_acc += compute_accuracy(predictions=predictions, labels=y_train)

                # Contrastive loss
                if self.contrastive_loss is not None:
                    for loss in self.contrastive_loss:
                        contrastive_element = self.get_contrastive_element(loss.name)
                        f0 = end_points_list[contrastive_element][0]
                        losses.append(loss(f0, end_points_list[contrastive_element][1:], y_train))

                # Init the grad to zeros first
                self.optimizer.zero_grad()

                # Backward your network
                loss = sum(losses)
                loss.backward()

                # Optimize the parameters
                self.optimizer.step()

            # Learning rate scheduler
            self.scheduler.step()

            # Validation
            val_loss, val_acc = self.validate()

            # Record learning history
            hist["acc"].append(train_acc / batch_count)
            hist["loss"].append(train_loss / batch_count)
            hist["val_acc"].append(val_acc)
            hist["val_loss"].append(val_loss)

        return hist

    def run(self,
            model,
            name,
            optimizer="adam",
            lr=1e-3,
            momentum=None,
            weight_decay=.0,
            loss="CrossEntropy",
            batch_size=128,
            epochs=200,
            corruption_mode=None,
            corruption_dist="uniform",
            orig_plus_aug=True,
            temperature=1.0,
            rand_aug=False):
        """
        Runs the benchmark

        # Arguments
            :param model: PyTorch model
            :param name: (str) model name
            :param optimizer: (str) name of the selected Keras optimizer
            :param lr: (float) learning rate
            :param momentum: (float) only relevant for the optimization algorithms that use momentum
            :param weight_decay: (float) only relevant for the optimization algorithms that use weight decay
            :param loss: (str | list) name of the selected Keras loss function(s)
            :param batch_size: (int) # of inputs in a mini-batch
            :param epochs: (int) # of full training passes
            :param corruption_mode: (str) requied for VisCo experiments
            :param corruption_dist: (str) requied for VisCo experiments
            :param orig_plus_aug: (bool) requied for VisCo experiments
            :param temperature: (float) temperature for contrastive distillation loss
            :param rand_aug: (bool) enable/disable random data augmentation

        :return: (history, score)
        """
        # Custom configurations
        self.model = model
        self.model.cuda()
        is_contrastive = False
        multi_class = False

        # Set the optimizer
        self.optimizer = self.get_optimizer(optimizer, lr=lr, momentum=momentum, weight_decay=weight_decay)

        # Set the learning rate scheduler
        self.scheduler = LR_Scheduler(lr, self.train_dataset, self.optimizer)

        # Set loss functions
        if type(loss) is list:
            self.classification_loss = self.get_loss_func(loss[0])
            self.contrastive_loss = []
            for i in range(1, len(loss)):
                self.contrastive_loss.append(self.get_loss_func(loss[i], temperature=temperature))
            is_contrastive = True
        else:
            self.classification_loss = self.get_loss_func(loss)
        if self.distillation:
            self.distillation_loss = self.get_loss_func("Distillation", temperature=temperature)

        # Training data generator
        corruption_mode = None if corruption_mode is None else corruption_mode.lower()
        if corruption_mode is None:
            self.trainGenerator = ImageGenerator(dataset=self.dataset,
                                                 batch_size=batch_size,
                                                 stage="train",
                                                 img_mean_mode=self.img_mean_mode,
                                                 seed=13)

        elif corruption_mode == "mixup":
            multi_class = True
            self.trainGenerator = MixUpGenerator(dataset=self.dataset,
                                                 batch_size=batch_size,
                                                 stage="train",
                                                 img_mean_mode=self.img_mean_mode,
                                                 seed=13,
                                                 orig_plus_aug=orig_plus_aug)

        elif corruption_mode == "cutout":
            self.trainGenerator = CutOutGenerator(dataset=self.dataset,
                                                  batch_size=batch_size,
                                                  stage="train",
                                                  img_mean_mode=self.img_mean_mode,
                                                  seed=13,
                                                  orig_plus_aug=orig_plus_aug)

        elif corruption_mode == "cutmix":
            multi_class = True
            self.trainGenerator = CutMixGenerator(dataset=self.dataset,
                                                  batch_size=batch_size,
                                                  stage="train",
                                                  img_mean_mode=self.img_mean_mode,
                                                  seed=13,
                                                  orig_plus_aug=orig_plus_aug)

        elif "augmix" == corruption_mode:
            self.trainGenerator = AugMixGenerator(dataset=self.dataset,
                                                  batch_size=batch_size,
                                                  stage="train",
                                                  img_mean_mode=self.img_mean_mode,
                                                  seed=13,
                                                  orig_plus_aug=orig_plus_aug)

        elif "randaugment" in corruption_mode:
            self.trainGenerator = RandAugmentGenerator(dataset=self.dataset,
                                                       batch_size=batch_size,
                                                       stage="train",
                                                       corruption_mode=corruption_mode,
                                                       img_mean_mode=self.img_mean_mode,
                                                       seed=13,
                                                       orig_plus_aug=orig_plus_aug)

        elif "acvc" == corruption_mode or "vc" == corruption_mode:
            self.trainGenerator = ACVCGenerator(dataset=self.dataset,
                                                batch_size=batch_size,
                                                stage="train",
                                                epochs=epochs,
                                                corruption_mode=corruption_mode,
                                                corruption_dist=corruption_dist,
                                                img_mean_mode=self.img_mean_mode,
                                                rand_aug=rand_aug,
                                                seed=13,
                                                orig_plus_aug=orig_plus_aug)

        else:
            self.trainGenerator = AblationGenerator(dataset=self.dataset,
                                                    batch_size=batch_size,
                                                    stage="train",
                                                    epochs=epochs,
                                                    corruption_mode=corruption_mode,
                                                    corruption_dist=corruption_dist,
                                                    img_mean_mode=self.img_mean_mode,
                                                    rand_aug=rand_aug,
                                                    seed=13,
                                                    orig_plus_aug=orig_plus_aug)

        # Validation data generator
        self.valGenerator = ImageGenerator(dataset=(self.x_test[self.train_dataset], self.y_test[self.train_dataset]),
                                           batch_size=batch_size,
                                           stage="test")

        # Train the model
        hist = self.train(epochs=epochs, multi_class=multi_class)

        # Evaluate the model with the test datasets
        log("-----------------------------------------------------------------------------")
        learning_mode = "contrastive" if is_contrastive else "vanilla"
        key = "%s[%s]" % (name, learning_mode)
        generalization_results = {}
        scores = []
        for test_set in self.x_test:
            score = self.test(self.x_test[test_set], self.y_test[test_set])
            generalization_results[test_set] = score
            log("%s %s Test accuracy: %.4f" % (name, test_set, score))

            # Avg. single-source domain generalization accuracy
            if test_set != self.train_dataset:
                scores.append(score)
        avg_score = float(np.mean(np.array(scores)))
        log("%s Avg. DG Test accuracy: %.4f" % (name, avg_score))

        if generalization_results == {}:
            generalization_results = None
        else:
            generalization_results["history"] = hist
            generalization_results = {key: generalization_results}
        self.record_generalization_results(generalization_results)
        log("-----------------------------------------------------------------------------")
        score = self.test(self.x_test[self.train_dataset], self.y_test[self.train_dataset])

        return hist, score