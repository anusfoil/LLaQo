# LLaQo: Towards a query-based coach in expressive performance assessment
[![arXiv Paper](https://img.shields.io/badge/arXiv-Paper-brightgreen)]() [![NeuroPiano-data](https://img.shields.io/badge/neuropiano-data-orange)](https://huggingface.co/datasets/anusfoil/NeuroPiano-data) 



### Environment

Our environment lam2 is downloadable from []. After downloading, simply do ```source /path/to/your/envs/lam2/bin/activate ```

checkpoints: please access from []. It contains:
- Vicuna-7b model: 
- our checkpoint: 
- audio encoder: 


### Inference

For the inference demo, 
```
python llaqo_chat.py --ckpt ./ckpts/checkpoint.pth --vicuna_dir ./ckpts/vicuna
```

### Datasets

For our new NeuroPiano-dataset, please refer to the [hf repository](https://huggingface.co/datasets/anusfoil/NeuroPiano-data) as well as its [analysis report](). For other datasets, please see the following 


### Training

To be updated

#### Acknowledgement

The codebase is adapted from the codebase of [APT]{}, which was originally adapted from the BLIP-2,  


#### Citaiton
```
@article{zhang2024llaqoassessment,
  title={{LLaQo: Towards a query-based coach in expressive performance assessment}},
  author={Zhang, Huan and Cheung, Vincent and Nishioka, Hayato and Dixon, Simon and Furuya, Shinichi},
  journal={arXiv preprint arXiv:},
  year={2024}
}
```

