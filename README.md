# GMG: A Video Prediction Method Based on Global  Focus and Motion Guided.
Official repository for GMG: A Video Prediction Method Based on Global  Focus and Motion Guided.


# Overview
We propose a novel approach to video prediction by introducing the Global Focus Module (GFM) and Motion Guided Module (MGM), achieving new state-of-the-art (SOTA) performance. This recurrent neural network-based method effectively predicts the motion of video subjects while capturing global data features more comprehensively. It demonstrates strong performance, particularly in handling complex spatiotemporal data, such as rainfall prediction. This advancement offers a new general video prediction solution and provides fresh insights into the development of recurrent neural networks in the field of video prediction.
![image](https://github.com/duyhlzu/GMG/blob/main/results/main%20structure.png)

# Visualization results
<div align="center">

| Moving MNIST | Moving FMNIST | 
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/mmnist_ConvLSTM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_fashionmnist_ConvLSTM.gif' height="auto" width="260" ></div> |

| Moving MNIST-CIFAR | KittiCaltech |
| :---: | :---: |
|  <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_mnist_cifar_ConvLSTM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kitticaltech_ConvLSTM.gif' height="auto" width="260" ></div> |

| KTH | Human 3.6M | 
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kth20_ConvLSTM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/human_ConvLSTM.gif' height="auto" width="260" ></div> |

| Traffic - in flow | Traffic - out flow |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-traffic/taxibj_in_flow_ConvLSTM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-traffic/taxibj_out_flow_ConvLSTM.gif' height="auto" width="260" ></div> |

| Weather - Temperature | Weather - Humidity |
|  :---: |  :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_temperature_5_625_ConvLSTM.gif' height="auto" width="360" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_humidity_5_625_ConvLSTM.gif' height="auto" width="360" ></div>|

| Weather - Latitude Wind | Weather - Cloud Cover | 
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_latitude_ConvLSTM.gif' height="auto" width="360" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_cloud_cover_5_625_ConvLSTM.gif' height="auto" width="360" ></div> |

| BAIR Robot Pushing | Kinetics-400 | 
| :---: | :---: |
| <div align=center><img src='https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/257872182-4f31928d-2ebc-4407-b2d4-1fe4a8da5837.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/257872560-00775edf-5773-478c-8836-f7aec461e209.gif' height="auto" width="260" ></div> |

</div>
