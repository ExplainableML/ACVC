# ACVC
> [**Attention Consistency on Visual Corruptions for Single-Source Domain Generalization**](https://arxiv.org),            
> [Ilke Cugu](https://cuguilke.github.io/), 
> [Massimiliano Mancini](https://www.eml-unitue.de/people/massimiliano-mancini), 
> [Yanbei Chen](https://www.eml-unitue.de/people/yanbei-chen), 
> [Zeynep Akata](https://www.eml-unitue.de/people/zeynep-akata)        
> *IEEE Computer Vision and Pattern Recognition Workshops (CVPRW), 2022* 

## Citation

If you use these codes in your research, please cite:

```bibtex

```
  
## Dependencies
- Prerequisites:
```
torch~=1.5.1+cu101
numpy~=1.19.5
torchvision~=0.6.1+cu101
Pillow~=8.3.1
matplotlib~=3.1.1
sklearn~=0.0
scikit-learn~=0.24.1
scipy~=1.6.1
imagecorruptions~=1.1.2
tqdm~=4.58.0
```

- We also include a YAML script `./acvc-pytorch.yml` that is prepared for an easy Anaconda environment setup. 

- One can also use the `./requirements.txt` if one knows one's craft.

## Training

Training is done via `./run.py`. To get the up-to-date list of commands:
```shell
python run.py --help
```

We include a sample script `./run_experiments.sh` for a quick start.

## Analysis

The benchmark results are prepared by `./analysis/GeneralizationExpProcessor.py`, which outputs LaTeX tables of the cumulative results in a .tex file.

- For example:
```
python GeneralizationExpProcessor.py --path generalization.json --to_dir ./results --image_format pdf
```

- You can also run distributed experiments, and merge the results later on:
```
python GeneralizationExpProcessor.py --merge_logs generalization_gpu0.json generalization_gpu1.json
```

## Case Study: COCO benchmark

COCO benchmark is especially useful for further studies on ACVC since it includes segmentation masks per image.

Here are the steps to make it work:
- TODO

## References

We indicate if a function or script is borrowed externally inside each file.
Specifically for visual corruption implementations we benefit from:

- The imagecorruptions library of [Autonomous Driving when Winter is Coming](https://github.com/bethgelab/imagecorruptions).

Consider citing this work as well if you use it in your project.
