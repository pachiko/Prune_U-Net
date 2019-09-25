# Taylor-Rank Pruning of U-Net via PyTorch
## Install
* ```tqdm```  
* ```torch```  
* ```numpy```  
* NO NEED for ```pydensecrf```  
## Usage
This performs ranking, removal, finetuning and evaluation in one pruning iteration.  
```python prune.py --load YOUR_MODEL.pth --channel_txt YOUR_CHANNELS.txt```
## Results

| Iteration  | Ranking Iterations | Pruned Channels | Finetuning Epochs/Iterations | Validation DICE | File Size (MB) |
| --- | --- | --- | --- | --- | --- |
| 0 | 500 | 300 | 0/1500 | 0.948 | 44.4 |
| 1 | 500 | 300 | 0/1500 | 0.861 | 38.9 |
| 2 | 500 | 300 | 0/1500 | 0.933 | 33.2 |
| 3 | 500 | 300 | 5/0 | 0.955 | 27.2 |

* Size Reduction: ```(52.4 – 27.2) / 52.4 x 100% = 48.1%```  
* Validation Dice Loss: ```98.53% – 95.5% = 3.03%```
## Enhancement
- [ ] Implement FLOPs Regularization
## Reference
### Dataset
https://www.kaggle.com/c/carvana-image-masking-challenge
### U-Net PyTorch Implementation
https://github.com/milesial/Pytorch-UNet
### U-Net Paper
https://arxiv.org/pdf/1505.04597.pdf
### Pruning Paper
https://arxiv.org/abs/1611.06440
