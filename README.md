<div align="center">

<h1>Let ViT Speak: Generative Language-Image Pre-training</h1>

<b>Yan Fang</b><sup>1,2,*</sup> · <b>Mengcheng Lan</b><sup>2,3,*</sup> · <b>Zilong Huang</b><sup>2,&dagger</sup> · <b>Weixian Lei<sup>2</sup><b> · <b>Yunqing Zhao<sup>2</sup><b> · <b>Yujie Zhong<sup>2</sup><b> · <b>Yingchen Yu<sup>2</sup><b> · <b>Qi She<sup>2</sup><b> · <b>Yao Zhao</b><sup>1</sup> · <b>Yunchao Wei</b><sup>1,&dagger</sup>

Beijing Jiaotong University<sup>1</sup> & Bytedance<sup>2</sup> & Nanyang Technological University<sup>3</sup>


<a href="https://huggingface.co/YanFang/GenLIP"><img src='https://img.shields.io/badge/Model-HuggingFace-orange' alt='Model HuggingFace'></a>
</div>

---

## Table of Contents
- [News](#news)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Datasets](#datasets)
  - [Configuration](#configuration)
- [Training](#training)
- [Model Checkpoints](#model-checkpoints)
- [Acknowledgments](#acknowledgments)

## News
- 2026-05-03: Code is ready to be released. [✔]

### Installation

```bash
# Clone the repository
git clone https://github.com/YanFangCS/GenLIP
cd GenLIP

# Install dependencies
pip install -r requirements.txt
```

### Datasets
We use several caption datasets in our pretraining process:

1. stage1
- [[Recap-DataComp-1B](https://huggingface.co/datasets/UCSC-VLAA/Recap-DataComp-1B)]: https://huggingface.co/datasets/UCSC-VLAA/Recap-DataComp-1B

2. stage2
- [[Infinity-MM](https://huggingface.co/datasets/BAAI/Infinity-MM)]: https://huggingface.co/datasets/BAAI/Infinity-MM/tree/main/stage1
- [[BLIP3o](BLIP3o/BLIP3o-Pretrain-Long-Caption)]: https://huggingface.co/datasets/BLIP3o/BLIP3o-Pretrain-Long-Caption

optional for stage2:
- [[CapRL](https://huggingface.co/datasets/internlm/CapRL-2M)]: https://huggingface.co/datasets/internlm/CapRL-2M
- [[PLM-Image-Auto](https://huggingface.co/datasets/facebook/PLM-Image-Auto)]: https://huggingface.co/datasets/facebook/PLM-Image-Auto (only capiton parts)

All these datasets need to be downloaded and sort into suitable formats for effective pretraining.

### Configuration

We provide three model configs in configs/model_configs/genlip:
- genlip_l16_224.json
- genlip_so16_224.json
- genlip_g16_224.json

Together with training configurations in configs/pretrain/genlip:
- stage1/train_genlip_*_recap.yaml
- stage2/train_genlip_*_navit.yaml

Remember to update the paths in the above config files to point to your local datasets before starting training.

## Training
A training script is provided in jobs/train.sh. You can start training with:
```bash
bash train.sh <main_func> <model_config>

# an example:
bash train.sh tasks/train_genlip_stage1.py configs/pretrain/genlip/stage1/train_genlip_so16_224_recap.yaml
```
where <main_func> is the main training script to be executed and <model_config> is the model configuration to be used.

All you need to do is to set the paths and appropriate hyperparameters in the config files, and wait for finishing the training process.

## Model Checkpoints

The models are available at https://huggingface.co/YanFang/GenLIP .


## Acknowledgments

Our codebase is built upon:
- [[VeOmni](https://github.com/ByteDance-Seed/VeOmni)]: https://github.com/ByteDance-Seed/VeOmni, a simple and high-performance multi-modal model training framework developed by ByteDance Seed team.

## 📄 License
This project is licensed under Apache License 2.0. See the `LICENSE` file for details.