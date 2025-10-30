# Importance-Weighted Training of amortised samplers

This repository contains the code for the paper "[Importance-Weighted Training of Diffusion Samplers](https://openreview.net/forum?id=lHkfoqd6YN)", presented at the GenBio Workshop @ ICML 2025. The paper introduces importance-weighted training scheme for diffusion samplers, which combines historical samples with adaptive importance weights so as to make the training samples better approximate the desired distribution. We use an off-policy training loss, Trajectory-Balance (TB), to train with the off-policy samples from the importance-weighted buffer. This repository extend the paper by incorporating biochemical sequence design tasks (in discrete spaces) with prepend/append models (c.f., [Shen et al., 2023](https://arxiv.org/abs/2305.07170)). This code is based on the [GFN-Diffusion](https://github.com/GFNOrg/gfn-diffusion) repository, with extensive refactorings and modifications to adapt to the importance-weighted training framework.

Note that this work is a prequel to "[Reinforced sequential Monte Carlo for amortised sampling](https://arxiv.org/abs/2510.11711)", which incorporates sequential Monte Carlo (SMC) in addition to the importance-weighted buffer.

## Setup
- python 3.11
- torch 2.8.0

See requirements.txt for other dependencies. 
```bash
pip install -r requirements.txt
```

## Usage

Check the README.md files in the `gfn-diffusion` and `gfn-discrete` directories for more details.

## Citation
If you find this work useful, please cite our [follow-up work](https://arxiv.org/abs/2510.11711) that contains everything in this repository:

```bibtex
@article{choi2025reinforced,
  title={Reinforced sequential {M}onte {C}arlo for amortised sampling},
  author={Choi, Sanghyeok and Mittal, Sarthak and Elvira, V{\'\i}ctor and Park, Jinkyoo and Malkin, Nikolay},
  journal={arXiv preprint arXiv:2510.11711},
  year={2025}
}
```