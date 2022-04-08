#!/bin/bash
echo ""
echo "                  ---------------------------------                   "
echo "                 |                                 |                  "
echo "----------------- Domain Generalization Experiments ------------------"
echo "                 |                                 |                  "
echo "                  ---------------------------------                   " 
echo ""
echo "----------------------------------------------------------------------"
for i in 1 2 3 4 5; do
	echo "COCO experiments..."
	for depth in 18; do
		for lr in 4e-3; do 
			for img_mean in imagenet; do

				# Baseline
				echo "Exp #"$i" [Baseline]: Training variant = {depth: "$depth", lr: "$lr", img_mean:"$img_mean"}"
				python run.py --model resnet --depth $depth --pretrained_weights imagenet --lr $lr --optimizer sgd --batch_size 128 --img_mean_mode $img_mean --epochs 30 --train_dataset COCO --test_datasets DomainNet:Real DomainNet:Infograph DomainNet:Clipart DomainNet:Painting DomainNet:Quickdraw DomainNet:Sketch --corruption_mode None --print_config > dump

				# MixUp experiments
				echo "Exp #"$i" [MixUp]: Training variant = {depth: "$depth", lr: "$lr", img_mean:"$img_mean"}"
				python run.py --model resnet --depth $depth --pretrained_weights imagenet --lr $lr --optimizer sgd --batch_size 128 --img_mean_mode $img_mean --epochs 30 --train_dataset COCO --test_datasets DomainNet:Real DomainNet:Infograph DomainNet:Clipart DomainNet:Painting DomainNet:Quickdraw DomainNet:Sketch --corruption_mode mixup --print_config > dump

				# CutOut experiments
				echo "Exp #"$i" [CutOut]: Training variant = {depth: "$depth", lr: "$lr", img_mean:"$img_mean"}"
				python run.py --model resnet --depth $depth --pretrained_weights imagenet --lr $lr --optimizer sgd --batch_size 128 --img_mean_mode $img_mean --epochs 30 --train_dataset COCO --test_datasets DomainNet:Real DomainNet:Infograph DomainNet:Clipart DomainNet:Painting DomainNet:Quickdraw DomainNet:Sketch --corruption_mode cutout --print_config > dump

				# CutMix experiments
				echo "Exp #"$i" [CutMix]: Training variant = {depth: "$depth", lr: "$lr", img_mean:"$img_mean"}"
				python run.py --model resnet --depth $depth --pretrained_weights imagenet --lr $lr --optimizer sgd --batch_size 128 --img_mean_mode $img_mean --epochs 30 --train_dataset COCO --test_datasets DomainNet:Real DomainNet:Infograph DomainNet:Clipart DomainNet:Painting DomainNet:Quickdraw DomainNet:Sketch --corruption_mode cutmix --print_config > dump

				# RandAugment experiments
				echo "Exp #"$i" [RandAugment]: Training variant = {depth: "$depth", lr: "$lr", img_mean:"$img_mean"}"
				python run.py --model resnet --depth $depth --pretrained_weights imagenet --lr $lr --optimizer sgd --batch_size 128 --img_mean_mode $img_mean --epochs 30 --train_dataset COCO --test_datasets DomainNet:Real DomainNet:Infograph DomainNet:Clipart DomainNet:Painting DomainNet:Quickdraw DomainNet:Sketch --corruption_mode randaugment --print_config > dump

				# AugMix experiments
				echo "Exp #"$i" [AugMix]: Training variant = {depth: "$depth", lr: "$lr", img_mean:"$img_mean"}"
				python run.py --model resnet --depth $depth --pretrained_weights imagenet --lr $lr --loss CrossEntropy JSDivergence --optimizer sgd --batch_size 128 --img_mean_mode $img_mean --epochs 30 --train_dataset COCO --test_datasets DomainNet:Real DomainNet:Infograph DomainNet:Clipart DomainNet:Painting DomainNet:Quickdraw DomainNet:Sketch --corruption_mode augmix --print_config > dump

				# VC experiments
				echo "Exp #"$i" [VC]: Training variant = {depth: "$depth", lr: "$lr", img_mean:"$img_mean"}"
				python run.py --model resnet --depth $depth --pretrained_weights imagenet --lr $lr --optimizer sgd --batch_size 128 --img_mean_mode $img_mean --epochs 30 --train_dataset COCO --test_datasets DomainNet:Real DomainNet:Infograph DomainNet:Clipart DomainNet:Painting DomainNet:Quickdraw DomainNet:Sketch --corruption_mode vc --print_config > dump

				# ACVC experiments
				echo "Exp #"$i" [ACVC]: Training variant = {depth: "$depth", lr: "$lr", img_mean:"$img_mean"}"
				python run.py --model resnet --depth $depth --pretrained_weights imagenet --lr $lr --loss CrossEntropy AttentionConsistency --optimizer sgd --batch_size 128 --img_mean_mode $img_mean --epochs 30 --train_dataset COCO --test_datasets DomainNet:Real DomainNet:Infograph DomainNet:Clipart DomainNet:Painting DomainNet:Quickdraw DomainNet:Sketch --corruption_mode acvc --print_config > dump
			
			done
		done
	done
	echo "PACS experiments..."
	for depth in 18; do
		for lr in 4e-3; do 
			for img_mean in imagenet; do

				# Baseline
				echo "Exp #"$i" [Baseline]: Training variant = {depth: "$depth", lr: "$lr", img_mean:"$img_mean"}"
				python run.py --model resnet --depth $depth --pretrained_weights imagenet --lr $lr --optimizer sgd --batch_size 128 --img_mean_mode $img_mean --epochs 30 --train_dataset PACS:Photo --test_datasets PACS:Art PACS:Cartoon PACS:Sketch --corruption_mode None --print_config > dump

				# MixUp experiments
				echo "Exp #"$i" [MixUp]: Training variant = {depth: "$depth", lr: "$lr", img_mean:"$img_mean"}"
				python run.py --model resnet --depth $depth --pretrained_weights imagenet --lr $lr --optimizer sgd --batch_size 128 --img_mean_mode $img_mean --epochs 30 --train_dataset PACS:Photo --test_datasets PACS:Art PACS:Cartoon PACS:Sketch --corruption_mode mixup --print_config > dump

				# CutOut experiments
				echo "Exp #"$i" [CutOut]: Training variant = {depth: "$depth", lr: "$lr", img_mean:"$img_mean"}"
				python run.py --model resnet --depth $depth --pretrained_weights imagenet --lr $lr --optimizer sgd --batch_size 128 --img_mean_mode $img_mean --epochs 30 --train_dataset PACS:Photo --test_datasets PACS:Art PACS:Cartoon PACS:Sketch --corruption_mode cutout --print_config > dump
				
				# CutMix experiments
				echo "Exp #"$i" [CutMix]: Training variant = {depth: "$depth", lr: "$lr", img_mean:"$img_mean"}"
				python run.py --model resnet --depth $depth --pretrained_weights imagenet --lr $lr --optimizer sgd --batch_size 128 --img_mean_mode $img_mean --epochs 30 --train_dataset PACS:Photo --test_datasets PACS:Art PACS:Cartoon PACS:Sketch --corruption_mode cutmix --print_config > dump
				
				# RandAugment experiments
				echo "Exp #"$i" [RandAugment]: Training variant = {depth: "$depth", lr: "$lr", img_mean:"$img_mean"}"
				python run.py --model resnet --depth $depth --pretrained_weights imagenet --lr $lr --optimizer sgd --batch_size 128 --img_mean_mode $img_mean --epochs 30 --train_dataset PACS:Photo --test_datasets PACS:Art PACS:Cartoon PACS:Sketch --corruption_mode randaugment --print_config > dump

				# AugMix experiments
				echo "Exp #"$i" [AugMix]: Training variant = {depth: "$depth", lr: "$lr", img_mean:"$img_mean"}"
				python run.py --model resnet --depth $depth --pretrained_weights imagenet --lr $lr --loss CrossEntropy JSDivergence --optimizer sgd --batch_size 128 --img_mean_mode $img_mean --epochs 30 --train_dataset PACS:Photo --test_datasets PACS:Art PACS:Cartoon PACS:Sketch --corruption_mode augmix --print_config > dump

				echo "Exp #"$i" [VC]: Training variant = {depth: "$depth", lr: "$lr", img_mean:"$img_mean"}"
				python run.py --model resnet --depth $depth --pretrained_weights imagenet --lr $lr --optimizer sgd --batch_size 128 --img_mean_mode $img_mean --epochs 30 --train_dataset PACS:Photo --test_datasets PACS:Art PACS:Cartoon PACS:Sketch --corruption_mode vc --print_config > dump

				echo "Exp #"$i" [ACVC]: Training variant = {depth: "$depth", lr: "$lr", img_mean:"$img_mean"}"
				python run.py --model resnet --depth $depth --pretrained_weights imagenet --lr $lr --loss CrossEntropy AttentionConsistency --optimizer sgd --batch_size 128 --img_mean_mode $img_mean --epochs 30 --train_dataset PACS:Photo --test_datasets PACS:Art PACS:Cartoon PACS:Sketch --corruption_mode acvc --print_config > dump
			
			done
		done
	done
done
echo "Done."
