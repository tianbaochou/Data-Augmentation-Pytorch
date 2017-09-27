# Data-Augmentation-Pytorch
Data-Augmentation example based on torchsample
---
---
## 1. Introduction.
This is an example which adopts [torchsample](https://github.com/ncullen93/torchsample)  package to implement data augmentation. This package provides many data augmentation methods such as rotation, zoom in or out.
## 2. Demo
The CIFAR-10 classification task is used to show how to utilize this package to implement data augmentation.
### 2.1. Run Demo:
```bash
python main.py
```
and 
```bash
python advanced_main.py
```
### 2.2. Implementation:
Standard method: (random horizontal flip data augmentation.)

```python
import torchvision.transforms as transforms
import torchsample as ts
train_tf= transforms.Compose([
            transforms.RandomHorizontalFlip(), # data augmentation: random horizontal flip
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
```

Adding rotation data augmentation:

```python
import torchvision.transforms as transforms
import torchsample as ts
train_tf= transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ts.transforms.Rotate(20), # data augmentation: rotation 
            ts.transforms.Rotate(-20), # data augmentation: rotation
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
```
### 2.3. Accuracy.
The final accuracy is shown as follows:

| Network        | Baselines           | Data Augmentation/(Rotation)  |
| ------------- |:-------------:| -----:|
| AlexNet (No pretrained)     | 

### 2.4. Convergence Curve.

## 3.  Acknowledgement.

The  [torchsample](https://github.com/ncullen93/torchsample)  is a very awesome package implemented by [Nick Cullen](https://github.com/ncullen93). 
