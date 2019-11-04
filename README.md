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

### Without FLOPs Regularization:  

| Iteration  | Ranking Iterations | Pruned Channels | Finetuning Epochs/Iterations | Validation DICE | File Size (MB) |
| --- | --- | --- | --- | --- | --- |
| 0 | N/A | N/A | N/A | 0.985 | 52.4 |
| 1 | 500 | 300 | 0/1500 | 0.948 | 44.4 |
| 2 | 500 | 300 | 0/1500 | 0.861 | 38.9 |
| 3 | 500 | 300 | 0/1500 | 0.933 | 33.2 |
| 4 | 500 | 300 | 5/0 | 0.955 | 27.2 |

* Size Reduction: ```(52.4 – 27.2) / 52.4 x 100% = 48.1%```  
* Validation Dice Loss: ```98.53% – 95.5% = 3.03%```

### With FLOPs Regularization (```strength=0.001```)

| Iteration  | Ranking Iterations | Pruned Channels | Finetuning Epochs/Iterations | Validation DICE | File Size (MB) |
| --- | --- | --- | --- | --- | --- |
| 0 | N/A | N/A | N/A | 0.985 | 52.4 |
| 1 | 500 | 300 | 0/1500 | 0.979 | 44.5 |
| 2 | 500 | 300 | 0/1500 | 0.972 | 39.1 |
| 3 | 500 | 300 | 0/1500 | 0.975 | 32.2 |
| 4 | 500 | 300 | 0/1500 | 0.957 | 26.2 |
| 5 | 500 | 300 | 0/1500 | 0.960 | 20.8 |

* Size Reduction: ```(52.4 – 20.8) / 52.4 x 100% = 60.3%```  
* Validation Dice Loss: ```98.53% – 96.0% = 2.53%```
## Channels After Pruning
![alt text](https://raw.githubusercontent.com/kcang2/Prune_U-Net/master/relative.png)
![alt text](https://raw.githubusercontent.com/kcang2/Prune_U-Net/master/total.png)
## Enhancement
- [X] Calculate FLOPs
- [X] Implement FLOPs Regularization
## Reference
### Dataset
https://www.kaggle.com/c/carvana-image-masking-challenge
### U-Net PyTorch Implementation
https://github.com/milesial/Pytorch-UNet
### U-Net Paper
https://arxiv.org/pdf/1505.04597.pdf
### Pruning Paper
https://arxiv.org/abs/1611.06440
