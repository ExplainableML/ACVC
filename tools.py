import os
import logging
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score
from time import localtime, strftime
from enum import Enum

logging.basicConfig(filename=os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment.log"), filemode="a", format="%(message)s", level=logging.INFO)

# Limit unwanted logging messages from packages
warnings.filterwarnings("ignore", category=DeprecationWarning)
matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.ERROR)

class LogType(Enum):
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3

def log(msg, log_type=LogType.INFO, to_file=True, to_stdout=True):
    msg = "%s %s" % (get_time(), msg)

    if to_stdout:
        print(msg)
    if to_file and log_type == LogType.DEBUG:
        logging.debug(msg)
    elif to_file and log_type == LogType.INFO:
        logging.info(msg)
    elif to_file and log_type == LogType.WARNING:
        logging.warning(msg)
    elif to_file and log_type == LogType.ERROR:
        logging.error(msg)

def log_config(config):
    log("Active Configuration:")
    log("--------------------")
    for key in config:
        residual = 24 - len(key)
        temp = ""
        while len(temp) < residual:
            temp += " "
        log("%s%s: %s" % (key, temp, config[key]))

def to_scientific(x):
    return "{:.0e}".format(x)

def get_time():
    return "[%s]" % strftime("%a, %d %b %Y %X", localtime())

def get_arch_name(arch, depth=""):
    name = "Unknown"

    if arch == "resnet":
        name = "ResNet%s" % depth

    return name

def plot_learning_curve(training_hist, chart_path):
    """
    Plots the learning curve of the given training history

    # Arguments
        :param training_hist: (hist.history) of keras.models.Model.fit
        :param chart_path: (String) file path for the output chart
    """
    is_ok = True

    # Error handler for missing values
    for key in ["acc", "loss", "val_acc", "val_loss"]:
        if key not in training_hist:
            is_ok = False

    if is_ok:
        # Starting building the learning curve graph
        fig, ax1 = plt.subplots(figsize=(14, 9))
        epoch_list = list(range(1, len(training_hist['acc']) + 1))

        # Plotting training and test losses
        train_loss, = ax1.plot(epoch_list, training_hist['loss'], color='red', alpha=.5)
        if "loss_std" in training_hist:
            ax1.fill_between(epoch_list,
                             training_hist['loss'] + training_hist['loss_std'],
                             training_hist['loss'] - training_hist['loss_std'],
                             color="red",
                             alpha=.3)
        val_loss, = ax1.plot(epoch_list, training_hist['val_loss'], linewidth=2, color='green')
        if "val_loss_std" in training_hist:
            ax1.fill_between(epoch_list,
                             training_hist['val_loss'] + training_hist['val_loss_std'],
                             training_hist['val_loss'] - training_hist['val_loss_std'],
                             color="green",
                             alpha=.3)
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')

        # Plotting test accuracy
        ax2 = ax1.twinx()
        train_accuracy, = ax2.plot(epoch_list, training_hist['acc'], linewidth=1, color='orange')
        if "acc_std" in training_hist:
            ax2.fill_between(epoch_list,
                             training_hist['acc'] + training_hist['acc_std'],
                             training_hist['acc'] - training_hist['acc_std'],
                             color="orange",
                             alpha=.3)
        val_accuracy, = ax2.plot(epoch_list, training_hist['val_acc'], linewidth=2, color='blue')
        if "val_acc_std" in training_hist:
            ax2.fill_between(epoch_list,
                             training_hist['val_acc'] + training_hist['val_acc_std'],
                             training_hist['val_acc'] - training_hist['val_acc_std'],
                             color="blue",
                             alpha=.3)
        ax2.set_ylim(bottom=0, top=1)
        ax2.set_ylabel('Accuracy')

        # Adding legend
        plt.legend([train_loss, val_loss, train_accuracy, val_accuracy], ['Training Loss', 'Validation Loss', 'Training Accuracy', 'Validation Accuracy'], loc=7, bbox_to_anchor=(1, 0.8))
        plt.title('Learning Curve')

        # Saving learning curve
        plt.savefig(chart_path)
        plt.close(fig)

def plot_confusion_matrix(y_test, y_preds, chart_path, n_classes, class_labels=None):
    class_labels = [""]*n_classes if class_labels is None else class_labels

    #Generate the normalized confusion matrix
    cm = confusion_matrix(y_test, y_preds)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig = plt.figure(figsize=(33, 26))
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, class_labels, rotation=30)
    plt.yticks(tick_marks, class_labels)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.1f'),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    # Saving learning curve
    plt.savefig(chart_path)
    plt.close(fig)

def resize_image(img, target_dim):
    new_img = img.resize(target_dim, Image.ANTIALIAS)
    return new_img

def shuffle_data_old(samples, labels, segmentation_masks=None, seed=13):
    num = len(labels)
    shuffle_index = np.random.RandomState(seed=seed).permutation(np.arange(num))
    shuffled_samples = samples[shuffle_index]
    shuffled_labels = labels[shuffle_index]
    shuffled_masks = None if segmentation_masks is None else segmentation_masks[shuffle_index]
    return shuffled_samples, shuffled_labels, shuffled_masks

def shuffle_data(samples, labels, teacher_logits=None, segmentation_masks=None, seed=13):
    np.random.seed(seed)
    random_state = np.random.get_state()
    np.random.shuffle(samples)
    np.random.set_state(random_state)
    np.random.shuffle(labels)

    if segmentation_masks is not None:
        np.random.set_state(random_state)
        np.random.shuffle(segmentation_masks)

    if teacher_logits is not None:
        np.random.set_state(random_state)
        np.random.shuffle(teacher_logits)

    return samples, labels, teacher_logits, segmentation_masks

def get_contrastive_loss(loss, orig_plus_corrupted=False):
    if type(loss) is list:
        return loss[1]

    else:
        if orig_plus_corrupted:
            return "orig_plus_corrupted"

        else:
            return "None"

def compute_accuracy(predictions, labels):
    if np.ndim(labels) == 2:
        y_true = np.argmax(labels, axis=-1)
    else:
        y_true = labels
    accuracy = accuracy_score(y_true=y_true, y_pred=np.argmax(predictions, axis=-1))
    return accuracy
