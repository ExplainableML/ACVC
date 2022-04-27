import torch
import argparse
import configparser
from models.ResNet import get_resnet
from testers.DomainGeneralization_tester import DomainGeneralization_tester
from tools import *

datasets_str = """
    Supported benchmarks:
        - PACS
        - COCO
        - DomainNet
"""

def create_config_file(config):
    # Default configurations
    config["DEFAULT"] = {"version": "1.0.1",
                         "model": "resnet",
                         "depth": 18,
                         "lr": 4e-3,
                         "batch_size": 128,
                         "epochs": 30,
                         "optimizer": "sgd",
                         "momentum": 0.9,
                         "temperature": 1.0,
                         "img_mean_mode": "imagenet",
                         "corruption_mode": "None",
                         "corruption_dist": "uniform",
                         "only_corrupted": False,
                         "loss": "CrossEntropy",
                         "train_dataset": "PACS:Photo",
                         "test_datasets": "None",
                         "print_config": True,
                         "data_dir": "../../datasets",
                         "first_run": False,
                         "model_dir": ".",
                         "save_model": False,
                         "knowledge_distillation": False,
                         "random_aug": False}

    with open("settings.ini", "w+") as config_file:
        config.write(config_file)

if __name__ == '__main__':

    # Dynamic parameters
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", help="selected neural net architecture", type=str)
    parser.add_argument("--depth", help="# of layers", type=int)
    parser.add_argument("--lr", help="learning rate", type=float)
    parser.add_argument("--batch_size", help="batch size (must be an even number!)", type=int)
    parser.add_argument("--epochs", help="# of epochs", type=int)
    parser.add_argument("--optimizer", help="optimization algorithm", type=str)
    parser.add_argument("--momentum", help="momentum (only relevant if the 'optimizer' algorithm is using it)", type=float)
    parser.add_argument("--weight_decay", help="L2 regularization penalty", type=float)
    parser.add_argument("--temperature", help="temperature for contrastive distillation loss", type=float)
    parser.add_argument("--pretrained_weights", help="imagenet | None", type=str)
    parser.add_argument("--img_mean_mode", help="image mean subtraction mode for dataset preprocessing, options: None | per_pixel | per_channel", type=str)
    parser.add_argument("--corruption_mode", help="visual corruptions on inputs as a data augmentation method", type=str)
    parser.add_argument("--corruption_dist", help="distribution from which the corruption rate is randomly sampled per image", type=str)
    parser.add_argument("--only_corrupted", help="when info loss applied, only the corrupted images will be in the mini-batch", action="store_true")
    parser.add_argument("--loss", help="loss function(s)", nargs="+")
    parser.add_argument("--train_dataset", help=datasets_str, type=str)
    parser.add_argument("--test_datasets", help="list of test sets for domain generalization experiments", nargs="+")
    parser.add_argument("--to_path", help="filepath to save models with custom names", type=str)
    parser.add_argument("--data_dir", help="filepath to save datasets", type=str)
    parser.add_argument("--first_run", help="to initiate COCO preprocessing", action="store_true")
    parser.add_argument("--model_dir", help="filepath to save models", type=str)
    parser.add_argument("--print_config", help="prints the active configurations", action="store_true")
    parser.add_argument("--save_model", help="to save the trained models", action="store_true")
    parser.add_argument("--knowledge_distillation", help="to disable batch normalization", action="store_true")
    parser.add_argument("--random_aug", help="to enable random data augmentation", action="store_true")
    args = vars(parser.parse_args())

    # Static parameters
    config = configparser.ConfigParser(allow_no_value=True)
    try:
        if not os.path.exists("settings.ini"):
            create_config_file(config)

        # Override the default values if specified
        config.read("settings.ini")
        temp = dict(config["DEFAULT"])
        temp.update({k: v for k, v in args.items() if v is not None})
        config.read_dict({"DEFAULT": temp})
        config = config["DEFAULT"]

        # Assign the active values
        version = config["version"]
        arch = config["model"].lower()
        depth = int(config["depth"])
        lr = float(config["lr"])
        batch_size = int(config["batch_size"])
        epochs = int(config["epochs"])
        optimizer = config["optimizer"]
        momentum = float(config["momentum"])
        weight_decay = float(config["weight_decay"]) if "weight_decay" in config else .0
        temperature = float(config["temperature"])
        pretrained_weights = config["pretrained_weights"] if "pretrained_weights" in config else None
        img_mean_mode = config["img_mean_mode"] if config["img_mean_mode"].lower() != "none" else None
        corruption_mode = config["corruption_mode"] if config["corruption_mode"].lower() != "none" else None
        corruption_dist = config["corruption_dist"]
        loss = config["loss"]
        train_dataset = config["train_dataset"]
        test_datasets = config["test_datasets"]
        to_path = config["to_path"] if "to_path" in config else None
        data_dir = config["data_dir"]
        model_dir = config["model_dir"]
        FIRST_RUN = config["first_run"]
        PRINT_CONFIG = config.getboolean("print_config")
        SAVE_MODEL = config.getboolean("save_model")
        KNOWLEDGE_DISTILLATION = config.getboolean("knowledge_distillation")
        ONLY_CORRUPTED = config.getboolean("only_corrupted")
        RANDOM_AUG = config.getboolean("random_aug")
        log("Configuration is completed.")
    except Exception as e:
        log("Error: " + str(e), LogType.ERROR)
        log("Configuration fault! New settings.ini is created. Restart the program.", LogType.ERROR)
        create_config_file(config)
        exit(1)

    # Process benchmark parameters
    log("Single-source domain generalization experiment...")

    # Process selected neural net
    if arch not in ["resnet"]:
        log("Nice try... but %s is not a supported neural net architecture!" % arch, LogType.ERROR)
        exit(1)

    # Process selected datasets for benchmarking
    datasets = ["COCO",
                "PACS:Photo",
                "FullDomainNet:Real"]
    # Dataset checker
    if train_dataset not in datasets:
        log("Nice try... but %s is not an allowed dataset!" % train_dataset, LogType.ERROR)
        exit(1)

    # Process selected test datasets for domain generalization
    if args["test_datasets"] is not None and len(args["test_datasets"]) > 0:
        supported_datasets = ["PACS:Art",
                              "PACS:Cartoon",
                              "PACS:Sketch",
                              "PACS:Photo",
                              "DomainNet:Real",
                              "DomainNet:Infograph",
                              "DomainNet:Clipart",
                              "DomainNet:Painting",
                              "DomainNet:Quickdraw",
                              "DomainNet:Sketch",
                              "FullDomainNet:Infograph",
                              "FullDomainNet:Clipart",
                              "FullDomainNet:Painting",
                              "FullDomainNet:Quickdraw",
                              "FullDomainNet:Sketch"]
        # Dataset checker
        for s in args["test_datasets"]:
            if s not in supported_datasets:
                log("Nice try... but %s is not an allowed dataset!" % s, LogType.ERROR)
                exit(1)

        # Handle specific dataset selections
        test_datasets = args["test_datasets"]
    elif test_datasets == "None":
        test_datasets = None
    else:
        test_datasets = [test_datasets]

    # Process loss function(s)
    if args["loss"] is not None and len(args["loss"]) > 0:
        if len(args["loss"]) == 1:
            loss = args["loss"][0]
        else:
            loss = args["loss"]

    # Process the mini-batch state
    orig_plus_aug = False if ONLY_CORRUPTED else True

    # Log the active configuration if needed
    if PRINT_CONFIG:
        log_config(config)

    # Prepare the benchmark
    tester = DomainGeneralization_tester(train_dataset=train_dataset,
                                         test_dataset=test_datasets,
                                         img_mean_mode=img_mean_mode,
                                         data_dir=data_dir,
                                         distillation=KNOWLEDGE_DISTILLATION,
                                         first_run=FIRST_RUN,
                                         wait=True)
    tester.activate() # manually trigger the dataset loader
    n_classes = tester.get_n_classes()

    # Build the baseline model
    model_name = "%s[%s][img_mean=%s][aug=%s]" % (get_arch_name(arch, depth), train_dataset, img_mean_mode, corruption_mode)
    #model_name = "%s[%s][img_mean=%s][aug=%s_T%s]" % (get_arch_name(arch, depth), train_dataset, img_mean_mode, corruption_mode, temperature)

    if arch == "resnet":
        model = get_resnet(depth, n_classes)

    log("Baseline model is ready.")

    # Train the baseline model
    log("Baseline model training...")
    hist, score = tester.run(model,
                             name=model_name,
                             optimizer=optimizer,
                             lr=lr,
                             momentum=momentum,
                             weight_decay=weight_decay,
                             loss=loss,
                             batch_size=batch_size,
                             epochs=epochs,
                             corruption_mode=corruption_mode,
                             corruption_dist=corruption_dist,
                             orig_plus_aug=orig_plus_aug,
                             temperature=temperature,
                             rand_aug=RANDOM_AUG)

    log("%s Test accuracy: %s" % (model_name, score))
    log("----------------------------------------------------------------")

    # Plot and save the learning curve
    chart_path = "%s_learning_curve.png" % model_name
    chart_path = chart_path.replace(":", "_")
    plot_learning_curve(hist, chart_path)

    # Save the baseline model & print its structure
    if SAVE_MODEL:
        if to_path is None:
            torch.save(model, os.path.join(model_dir, "%s.pth" % model_name))
        else:
            torch.save(model, to_path)
        print(model)

    del model

    log("Done.")
