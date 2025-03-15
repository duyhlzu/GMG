# GMG: A Video Prediction Method Based on Global  Focus and Motion Guided.
<p align="left">
<a href="https://github.com/duyhlzu/GMG/blob/main/LICENSE" alt="license">
    <img src="https://img.shields.io/badge/license-Apache--2.0-%23002FA7" /></a>
</p>
Official repository for GMG: A Video Prediction Method Based on Global  Focus and Motion Guided. [Paper Link Coming Soon]

## Overview
We propose a novel approach to video prediction by introducing the Global Focus Module (GFM) and Motion Guided Module (MGM), achieving new state-of-the-art (SOTA) performance. This recurrent neural network-based method effectively predicts the motion of video subjects while capturing global data features more comprehensively. It demonstrates strong performance, particularly in handling complex spatiotemporal data, such as rainfall prediction. This advancement offers a new general video prediction solution and provides fresh insights into the development of recurrent neural networks in the field of video prediction.
![image](https://github.com/duyhlzu/GMG/blob/main/results/main%20structure.png)

## Visualization results
<div align="center">

| CIKM2017 | Shanghai2020 | 
| :---: | :---: |
| <div align=center><img src='https://github.com/duyhlzu/GMG/blob/main/results/comparison_CIKM.gif' height="auto" width="450" ></div> | <div align=center><img src='https://github.com/duyhlzu/GMG/blob/main/results/comparison_shanghai.gif' height="auto" width="450" ></div> |

| Moving MNIST | Taxibj |
| :---: | :---: |
|  <div align=center><img src='https://github.com/duyhlzu/GMG/blob/main/results/comparison_movingMNIST.gif' height="auto" width="450" ></div> | <div align=center><img src='https://github.com/duyhlzu/GMG/blob/main/results/comparison_taxibj.gif' height="auto" width="450" ></div> |

</div>

## Get Start
Our code is based on [OpenSTL](https://github.com/chengtan9907/OpenSTL). You need to migrate the model code to the OpenSTL framework to ensure a fairer comparison.
- `configs/:` You can conduct experiments by adding the configs used by GMG.
- `openstl/methods:` Contains defined training method of GMG. You need to add the file `gsamotionrnn.py` and replace the original `__init__.py`.
- `openstl/models:` The basic structure code for GMG is included. You need to add the file `gsamotionrnn_model.py` and replace the `original __init__.py`.
- `openstl/modules:` The main structure of GMG includes GMG-s, GMG-m, and GMG-L. The `gsamotion_modules.py` file contains the code for ST-ConvLSTM, Global Focus Module, and Self-Attention Memory. The code for the Motion Guided Module is included in `modules/layers/MotionGRU.py`. You need to add these files and replace the original `__init__.py` (in both directories).
- `openstl/api:` For the model's experimental setup, you need to replace the original `__init__.py`.
- `openstl/utils:` Please replace `parser.py`, as it pertains to the recognition of experiment command inputs.

*Note:* Since OpenSTL does not include the MotionRNN code by default, we have provided the corresponding code. You can follow the same steps as above to add it. The original code for MotionRNN can be referenced from the work of [Wang Yunbo et al](https://github.com/thuml/MotionRNN).

## Train
```
# Moving MNIST
python tools/train.py -d mmnist --epoch 600 -c configs/mmnist/GMG.py --ex_name mmnist_GMG
# TaxiBJ
python tools/train.py -d taxibj --epoch 200 -c configs/taxibj/GMG.py --ex_name taxibj_GMG
# WeatherBench
python tools/train.py --gpus 0 1 2 3 -d weather --epoch 100 -c configs/weather/t2m_5_625/GMG.py --ex_name weather_GMG
```

### Quantitative Results in TaxiBJ Dataset

| Model                  | MSE × 100 (↓) | MAE (↓)   | SSIM (↑)  | PSNR (↑)  |
|------------------------|---------------|-----------|-----------|-----------|
| ConvLSTM         | 40.0678       | 16.1892   | 0.9819    | 38.888    |
| PredRNN            | 35.0348       | 15.1302   | 0.9844    | 39.591    |
| PredRNN++         | 41.8227       | 15.9766   | 0.9824    | 39.135    |
| PyDNet           | 40.1700       | 16.4790   | 0.9808    | 38.939    |
| MotionRNN         | 37.6517       | 16.0009   | 0.9825    | 39.001    |
| MAU              | 40.7206       | 15.6620   | 0.9822    | 39.353    |
| PredRNN-V2       | 45.2737       | 16.6105   | 0.9807    | 38.713    |
| SimVP-gSTA         | 36.7385       | 15.3530   | 0.9832    | 39.509    |
| Swin-LSTM      | 35.9456       | 15.2276   | 0.9832    | 39.645    |
| TAU              | 35.1037       | 15.1745   | 0.9838    | 39.593    |
| WasT            | **29.7753**       | 14.7945   | 0.9846    | 39.777    |
| **GMG (Ours)**         | 29.8812   | **14.7277**| **0.9850**| **39.831**|

### Quantitative Results in WeatherBench Dataset

| Model                  | MSE (↓)  | MAE (↓)  | RMSE (↓) |
|------------------------|----------|----------|----------|
| ConvLSTM               | 1.9703   | 0.8472   | 1.4036   |
| PredRNN                | 1.2881   | 0.7035   | 1.1349   |
| MIM                    | 1.8480   | 0.8611   | 1.3594   |
| MotionRNN              | 1.2607   | 0.6813   | 1.1228   |
| MAU                    | 1.4381   | 0.7548   | 1.1992   |
| PredRNN-V2             | 1.8430   | 0.9029   | 1.3575   |
| SimVP-gSTA             | 1.5934   | 0.7654   | 1.2623   |
| TAU                    | 1.4986   | 0.7375   | 1.2241   |
| WaST                   | 1.3387   | 0.6808   | 1.1570   |
| **GMG (Ours)**         | **1.2341**| **0.6780**| **1.1109**|

### Quantitative Results in Moving MNIST Dataset

| Model                  | MSE (↓)  | MAE (↓)  | SSIM (↑) | PSNR (↑) |
|------------------------|----------|----------|----------|----------|
| SimVP-gSTA             | 22.5268  | 67.8671  | 0.9500   | 23.6783  |
| SimVP-ViT              | 26.4819  | 77.5663  | 0.9371   | 23.0279  |
| SimVP-VAN              | 20.5918  | 63.5674  | 0.9547   | 24.1267  |
| SimVP-Poolformer      | 25.5146  | 74.6528  | 0.9429   | 23.1193  |
| Swin-LSTM              | 19.4554  | 61.2669  | 0.9571   | 24.3593  |
| TAU                    | 19.9112  | 62.1182  | 0.9562   | 24.3096  |
| WaST                   | 22.0719  | 70.8779  | 0.9491   | 23.7451  |
| **GMG (Ours)**         | **19.0741**| **60.7413**| **0.9586**| **24.4606**|

## Acknowledgments

Our code is based on [OpenSTL](https://github.com/chengtan9907/OpenSTL) and [MotionRNN](https://github.com/thuml/MotionRNN). We greatly appreciate the code base they provided for this project.

## Citation

If you find this repository useful, please consider citing our paper:
[Coming Soon]
