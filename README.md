# SSIAI_VectorSpaces

Official code for <a href="https://arxiv.org/abs/2402.00261">Understanding Neural Network Systems for Image Analysis using Vector Spaces and Inverse Maps</a>:

In this paper, we introduce techniques from Linear Algebra to model neural network layers as maps between signal spaces. First, we demonstrate how signal spaces can be used to visualize weight spaces and convolutional layer kernels. We also demonstrate how residual vector spaces can be used to further visualize information lost at each layer. Second, we study invertible networks using vector spaces for computing input images that yield specific outputs. We demonstrate our approach on two invertible networks and ResNet18.

<img width="1016" height="364" alt="Screenshot 2025-12-01 at 1 28 52 PM" src="https://github.com/user-attachments/assets/c015d19e-a6ea-4a2e-91f2-d7291ae338f7" />

## Contents

```bash
├── src
│   ├── DataHelper.py
│   ├── LayerVisualizer.py
│   ├── TrainerHelper.py
├── models
│   ├── final_model_resnet
├── demo
│   ├── SSIAI_Demo.ipynb
├── README.md
```

## Citing

```
@inproceedings{pattichis2024understanding,
  title={Understanding Neural Network Systems for Image Analysis Using Vector Spaces},
  author={Pattichis, Rebecca and Pattichis, Marios S},
  booktitle={2024 IEEE Southwest Symposium on Image Analysis and Interpretation (SSIAI)},
  pages={73--76},
  year={2024},
  organization={IEEE}
}
```
