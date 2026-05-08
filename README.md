<div align="center">

<h1>Let ViT Speak: Generative Language-Image Pre-training</h1>

<b>Yan Fang</b><sup>1,2,&#42;</sup> · <b><a href="https://mc-lan.github.io">Mengcheng Lan</a></b><sup>2,3,&#42;</sup> · <b><a href="https://speedinghzl.github.io">Zilong Huang</a></b><sup>2,&dagger;</sup> · <b>Weixian Lei</b><sup>2</sup> · <b><a href="https://yunqing-me.github.io">Yunqing Zhao</a></b><sup>2</sup> · <b><a href="https://y-zhong.info">Yujie Zhong</a></b><sup>2</sup> · <b><a href="https://yingchen001.github.io">Yingchen Yu</a></b><sup>2</sup> · <b><a href="https://qi-she.net/">Qi She</a></b><sup>2</sup> · <b>Yao Zhao</b><sup>1</sup> · <b><a href="https://weiyc.github.io">Yunchao Wei</a></b><sup>1,&dagger;</sup>

Beijing Jiaotong University<sup>1</sup> & Bytedance<sup>2</sup> & Nanyang Technological University<sup>3</sup>

<a href="https://yanfangcs.github.io/vitspeak"><img src="https://img.shields.io/badge/Github-Page-blue" alt="Home Page"></a>
<a href="https://arxiv.org/abs/2605.00809"><img src="https://img.shields.io/badge/Paper-Arxiv-red" alt="Paper Arxiv"></a>
<a href="https://huggingface.co/collections/YanFang/genlip"><img src="https://img.shields.io/badge/Model-HuggingFace-orange" alt="Model HuggingFace"></a>
</div>

**TL;DR:** **GenLIP - lets ViT speak.** We show that a strong MLLM vision encoder can be pretrained with just **one Transformer** and **one autoregressive language modeling objective**—no contrastive loss, no dual-tower architecture, and no extra text decoder. Despite its simplicity, GenLIP scales effectively and performs well as a vision encoder in MLLMs, with particularly strong gains on Doc&OCR tasks.

<div align='center'>
  <img src="assets/teaser.png" alt="teaser" style="height: 200px; width: auto;">
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
- [[PLM-Image-Auto](https://huggingface.co/datasets/facebook/PLM-Image-Auto)]: https://huggingface.co/datasets/facebook/PLM-Image-Auto (only caption parts)

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

The models are available at [[huggingface](https://huggingface.co/collections/YanFang/genlip)].


## Acknowledgments

Our codebase is built upon:
- [[VeOmni](https://github.com/ByteDance-Seed/VeOmni)]: https://github.com/ByteDance-Seed/VeOmni, a simple and high-performance multi-modal model training framework developed by ByteDance Seed team.

## 📄 License
This project is licensed under Apache License 2.0. See the `LICENSE` file for details.


## Citation and Acknowledgement
If you find this project helpful, please give us a star ⭐ and cite our [paper](https://arxiv.org/pdf/2605.00809):

```bibtex
@article{fang2026letvitspeakgenerative,
  title={Let ViT Speak: Generative Language-Image Pre-training}, 
  author={Yan Fang and Mengcheng Lan and Zilong Huang and Weixian Lei and Yunqing Zhao and Yujie Zhong and Yingchen Yu and Qi She and Yao Zhao and Yunchao Wei},
  journal={arXiv preprint arXiv:2605.00809},
  year={2026}
}
```