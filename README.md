# GMG: A Video Prediction Method Based on Global  Focus and Motion Guided.
<p align="left">
<a href="https://github.com/duyhlzu/GMG/blob/main/LICENSE" alt="license">
    <img src="https://img.shields.io/badge/license-Apache--2.0-%23002FA7" /></a>
<a href="https://img.shields.io/github/stars/duyhlzu/GMG" alt="arXiv">
    <img src="https://img.shields.io/github/stars/duyhlzu/GMGL" /></a>
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

### Quantitative Results in CIKM2017 Dataset


| Model                          | Params | MSE(↓)  | MAE(↓)  | SSIM(↑)  | CSI30(↑) | CSI40(↑) | CSI50(↑) |
|--------------------------------|--------|---------|---------|---------|---------|---------|---------|
| ConvLSTM (NeurIPS’2015)        | 15.1M  | 28.8162 | 161.242 | 0.7359  | 0.7757  | 0.6590  | 0.5367  |
| PredRNN (NeurIPS’2017)         | 23.8M  | 28.8126 | 159.404 | 0.7303  | 0.7803  | 0.6690  | 0.5471  |
| PredRNN++ (ICML’2018)          | 38.6M  | 28.0987 | 159.258 | 0.7362  | 0.7775  | 0.6684  | 0.5533  |
| MIM (CVPR’2019)                | 38.04M | 27.2072 | 154.269 | 0.7388  | 0.7828  | 0.6725  | 0.5587  |
| PyDNet (CVPR’2020)             | 3.1M   | 28.9757 | 161.143 | 0.7393  | 0.7809  | 0.6654  | 0.5489  |
| MotionRNN (CVPR’2021)          | 26.9M  | 27.2091 | 155.827 | 0.7406  | 0.7867  | 0.6762  | 0.5510  |
| MAU (NeurIPS’2021)             | 4.48M  | 30.9094 | 167.361 | 0.7234  | 0.7731  | 0.6584  | 0.5459  |
| PredRNN-V2 (TPAMI’2022)        | 23.9M  | 28.0085 | 161.508 | 0.7374  | 0.7857  | 0.6640  | 0.5362  |
| SimVP-gSTA (CVPR’2022)         | 4.82M  | 31.3121 | 166.261 | 0.7195  | 0.7731  | 0.6594  | 0.5416  |
| SimVP-ViT (NeurIPS’2023)       | 39.6M  | 27.8803 | 157.478 | 0.7358  | 0.7835  | 0.6669  | 0.5483  |
| Swin-LSTM (ICCV’2023)          | 20.19M | 27.9612 | 158.899 | 0.7405  | 0.7826  | 0.6652  | 0.5532  |
| TAU (CVPR’2023)                | 38.4M  | 30.5827 | 161.948 | 0.7277  | 0.7779  | 0.6601  | 0.5370  |
| WaST (AAAI’2024)               | 28.18M | 30.3074 | 165.804 | 0.7309  | 0.7773  | 0.6534  | 0.5174  |
| **GMG (Ours)**                 | 31.44M | **25.0215** | **149.493** | **0.7513**  | **0.7885**  | **0.6812**  | **0.5682**  |

### Quantitative Results in Shanghai2020 Dataset

| Model                          | Params | MSE(↓)  | MAE(↓)  | SSIM(↑)  | CSI30(↑) | CSI40(↑) | CSI50(↑) |
|--------------------------------|--------|---------|---------|---------|---------|---------|---------|
| ConvLSTM (NeurIPS’2015)        | 15.1M  | 5.0219  | 39.236  | 0.9162  | 0.4321  | 0.4046  | 0.3584  |
| PredRNN (NeurIPS’2017)         | 23.8M  | 4.3347  | **34.297**  | 0.9288  | 0.4707  | 0.4451  | 0.3992  |
| PredRNN++ (ICML’2018)          | 38.6M  | 4.7445  | 39.416  | 0.9190  | 0.4419  | 0.4100  | 0.3605  |
| MIM (CVPR’2019)                | 38.04M | 6.3924  | 44.986  | 0.8997  | 0.3910  | 0.3595  | 0.3131  |
| PyDNet (CVPR’2020)             | 3.1M   | 7.6126  | 50.155  | 0.8846  | 0.3508  | 0.3124  | 0.2663  |
| MotionRNN (CVPR’2021)          | 26.9M  | 4.5867  | 37.871  | 0.9221  | 0.4457  | 0.4183  | 0.3708  |
| MAU (NeurIPS’2021)             | 4.48M  | 7.3441  | 50.766  | 0.8853  | 0.3485  | 0.3098  | 0.2599  |
| PredRNN-V2 (TPAMI’2022)        | 23.9M  | 4.2050  | 35.015  | 0.9270  | 0.4543  | 0.4272  | 0.3815  |
| SimVP-gSTA (CVPR’2022)         | 4.82M  | 8.0889  | 45.846  | 0.8908  | 0.3649  | 0.3406  | 0.2991  |
| SimVP-ViT (NeurIPS’2023)       | 39.6M  | 9.9074  | 52.900  | 0.8718  | 0.3272  | 0.3053  | 0.2667  |
| Swin-LSTM (ICCV’2023)          | 20.19M | 6.7183  | 46.025  | 0.8957  | 0.3714  | 0.3371  | 0.2902  |
| TAU (CVPR’2023)                | 38.4M  | 8.2874  | 47.315  | 0.8886  | 0.3630  | 0.3421  | 0.3000  |
| WaST (AAAI’2024)               | 28.18M | 6.1937  | 42.387  | 0.9063  | 0.3908  | 0.3545  | 0.3042  |
| **GMG (Ours)**                 | 31.44M | **4.0308**  | 35.771  | **0.9300**  | **0.4741**  | **0.4487**  | **0.4002**  |

### Quantitative Results in TaxiBJ Dataset

| Model                          | MSE × 100 (↓) | MAE (↓)   | SSIM (↑)  | PSNR (↑)  |
|--------------------------------|---------------|-----------|-----------|-----------|
| ConvLSTM (NeurIPS’2015)        | 40.0678       | 16.1892   | 0.9819    | 38.888    |
| PredRNN (NeurIPS’2017)         | 35.0348       | 15.1302   | 0.9844    | 39.591    |
| PredRNN++ (ICML’2018)          | 41.8227       | 15.9766   | 0.9824    | 39.135    |
| PyDNet (CVPR’2020)             | 40.1700       | 16.4790   | 0.9808    | 38.939    |
| MotionRNN (CVPR’2021)          | 37.6517       | 16.0009   | 0.9825    | 39.001    |
| MAU (NeurIPS’2021)             | 40.7206       | 15.6620   | 0.9822    | 39.353    |
| PredRNN-V2 (TPAMI’2022)        | 45.2737       | 16.6105   | 0.9807    | 38.713    |
| SimVP-gSTA (CVPR’2022)         | 36.7385       | 15.3530   | 0.9832    | 39.509    |
| Swin-LSTM (ICCV’2023)          | 35.9456       | 15.2276   | 0.9832    | 39.645    |
| TAU (CVPR’2023)                | 35.1037       | 15.1745   | 0.9838    | 39.593    |
| WaST (AAAI’2024)               | **29.7753**   | 14.7945   | 0.9846    | 39.777    |
| **GMG (Ours)**                 | 29.8812       | **14.7277**| **0.9850**| **39.831**|


### Quantitative Results in WeatherBench Dataset

| Model                          | MSE (↓)  | MAE (↓)  | RMSE (↓) |
|--------------------------------|----------|----------|----------|
| ConvLSTM (NeurIPS’2015)        | 1.9703   | 0.8472   | 1.4036   |
| PredRNN (NeurIPS’2017)         | 1.2881   | 0.7035   | 1.1349   |
| MIM (CVPR’2019)                | 1.8480   | 0.8611   | 1.3594   |
| MotionRNN (CVPR’2021)          | 1.2607   | 0.6813   | 1.1228   |
| MAU (NeurIPS’2021)             | 1.4381   | 0.7548   | 1.1992   |
| PredRNN-V2 (TPAMI’2022)        | 1.8430   | 0.9029   | 1.3575   |
| SimVP-gSTA (CVPR’2022)         | 1.5934   | 0.7654   | 1.2623   |
| TAU (CVPR’2023)                | 1.4986   | 0.7375   | 1.2241   |
| WaST (AAAI’2024)               | 1.3387   | 0.6808   | 1.1570   |
| **GMG (Ours)**                 | **1.2341**| **0.6780**| **1.1109**|


### Quantitative Results in Moving MNIST Dataset

| Model                          | MSE (↓)  | MAE (↓)  | SSIM (↑) | PSNR (↑) |
|--------------------------------|----------|----------|----------|----------|
| SimVP-gSTA (CVPR’2022)         | 22.5268  | 67.8671  | 0.9500   | 23.6783  |
| SimVP-ViT (CVPR’2022)          | 26.4819  | 77.5663  | 0.9371   | 23.0279  |
| SimVP-VAN (CVPR’2022)          | 20.5918  | 63.5674  | 0.9547   | 24.1267  |
| SimVP-Poolformer (CVPR’2022)   | 25.5146  | 74.6528  | 0.9429   | 23.1193  |
| Swin-LSTM (ICCV’2023)          | 19.4554  | 61.2669  | 0.9571   | 24.3593  |
| TAU (CVPR’2023)                | 19.9112  | 62.1182  | 0.9562   | 24.3096  |
| WaST (AAAI’2024)               | 22.0719  | 70.8779  | 0.9491   | 23.7451  |
| **GMG (Ours)**                 | **19.0741**| **60.7413**| **0.9586**| **24.4606**|

## Acknowledgments

Our code is based on [OpenSTL](https://github.com/chengtan9907/OpenSTL) and [MotionRNN](https://github.com/thuml/MotionRNN). We greatly appreciate the code base they provided for this project.

## Citation

If you find this repository useful, please consider citing our paper:
[Coming Soon]
