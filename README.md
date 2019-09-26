# Taylor-Rank Pruning of U-Net via PyTorch
## Requirements
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
| 0 | N/A | N/A | N/A | 0.985 | 52.4 |
| 1 | 500 | 300 | 0/1500 | 0.948 | 44.4 |
| 2 | 500 | 300 | 0/1500 | 0.861 | 38.9 |
| 3 | 500 | 300 | 0/1500 | 0.933 | 33.2 |
| 4 | 500 | 300 | 5/0 | 0.955 | 27.2 |

* Size Reduction: ```(52.4 – 27.2) / 52.4 x 100% = 48.1%```  
* Validation Dice Loss: ```98.53% – 95.5% = 3.03%```
## Channels

| Layer | Before Pruning | After Pruning | Channels Removed | Relative % of Channels Removed | % of Total Channels Removed |
| --- | --- | --- | --- | --- | --- |
| 0 | 64 | 13 | 51 | 79.7 | 2.9 |
| 1 | 64 | 19 | 45 | 70.3 | 2.6 |
| 2 | 128 | 74 | 54 | 42.2 | 3.1 |
| 3 | 128 | 68 | 60 | 46.9 | 3.4 |
| 4 | 256 | 182 | 74 | 28.9 | 4.2 |
| 5 | 256 | 190 | 66 | 25.8 | 3.8 |
| 6 | 512 | 393 | 119 | 23.2 | 6.8 |
| 7 | 512 | 361 | 151 | 29.5 | 8.6 |
| 8 | 512 | 373 | 139 | 27.1 | 7.9 |
| 9 | 512 | 388 | 124 | 24.2 | 7.0 |
| 10 | 1024 | 749 | 275 | 26.9 | 15.6 |
| 11 | 256 | 187 | 69 | 27.0 | 3.9 |
| 12 | 256 | 189 | 67 | 26.2 | 3.8 |
| 13 | 512 | 379 | 133 | 26.0 | 7.6 |
| 14 | 128 | 73 | 55 | 43.0 | 3.1 |
| 15 | 128 | 97 | 31 | 24.2 | 1.8 |
| 16 | 256 | 165 | 91 | 35.5 | 5.2 |
| 17 | 64 | 38 | 26 | 40.6 | 1.5 |
| 18 | 64 | 49 | 15 | 23.4 | 0.9 |
| 19 | 128 | 68 | 60 | 46.9 | 3.4 |
| 20 | 64 | 18 | 46 | 71.9 | 2.6 |
| 21 | 64 | 56 | 8 | 12.5 | 0.5 |  

![alt text](https://imgur.com/a/iv7w2NV)
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
