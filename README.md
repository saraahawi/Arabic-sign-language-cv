# Real-Time Arabic Sign Language Interpreter

## Problem Statement: 
Effective communication for deaf and mute people is very hard when they meet people who don’t know or speak sign language. Pattern recognition and computer vision can be used to mitigate that issue and make communication easier for those minorities. This projects aims to use transfer learning to recognize hand gestures and interpret their meaning and which letter they refer to in order to help deaf and mute people communicate with us and vice versa.

## Data
The [ArSL2018](https://data.mendeley.com/datasets/y7pckrw6z2/1) is a new comprehensive fully labelled dataset of Arabic Sign Language images launched in Prince Mohammad Bin Fahd University, Al Khobar, Saudi Arabia to be made available for researchers in the field of Machine Learning and Deep Learning. The ArSL2018 dataset is compiled of 54,049 images in gray scale with 64 × 64 dimension. Variations of images were introduced with different lighting and different background. 

![](https://els-jbs-prod-cdn.jbs.elsevierhealth.com/cms/attachment/47d41d2b-27e6-48d3-bce5-8705837d36ae/gr1_lrg.jpg)

| # | Letter name in English Script | Letter name in Arabic Script | # of Images |
:-------------------------:|:-------------------------: | :-------------------------: | :-------------------------:
| 1 | Alif | أَلِف)أ) | 1672
| 2 | Bā | بَاء) ب)| 1791
| 3 | Tā | أتَاء) ت) | 1838
| 4 | Thā | ثَاء) ث) | 1766
| 5 | Jīm | جِيمْ) ج) | 1552
| 6 | Hā | حَاء) ح) | 1526
| 7 | Khā | خَاء) خ) | 1607
| 8 | Dāl | دَالْ) د) | 1634
| 9 | Dhāl | ذَال) ذ) | 1582
| 10 | Rā | رَاء) ر) | 1659
| 11 | Zāy | زَاي) ز) | 1374
| 12 | Sīn | سِينْ) س) | 1638
| 13 | Shīn | شِينْ) ش) | 1507
| 14 | Sād | صَادْ)ص) | 1895
| 15 | Dād | ضَاد)ض) | 1670
| 16 | Tā | طَاء)ط) | 1816
| 17 | Zā | ظَاء)ظ) | 1723
| 18 | Ayn | عَين)ع) | 2114
| 19 | Ghayn | غَين)غ) | 1977
| 20 | Fā | فَاء)ف) | 1955
| 21 | Qāf | قَاف) ق) | 1705
| 22 | Kāf | كَاف)ك) | 1774
| 23 | Lām | لاَمْ)ل) | 1832
| 24 | Mīm | مِيمْ)م) | 1765
| 25 | Nūn | نُون)ن) | 1819
| 26 | Hā | هَاء)ه) | 1592
| 27 | Wāw | وَاو)و) | 1371
| 28 | Yā | يَا) ئ) | 1722
| 29 | Tāa | ة)ة) | 1791
| 30 | Al | ال)ال) | 1343
| 31 | Laa | ﻻ)ﻻ) | 1746
| 32 | Yāa | يَاء) يَاء)| 1293

## Steps: 
1) Labeling images for object detection.
2) Generating TensorFlow Records. 
3) Training transfer learning model on my data.
4) Fin-tuning my model to improve its performance.
4) Test and deploy model.

## Take-home message:
Transfer learning is a powerful tool used to build highly capable models, however, many factors affect the transfer learning model performance. The images of the dataset used in this work are of low-quality and inadequate variation, in fact many of them look identicel. Training the model on images that are of better quality and greater variation will improve the performance of the transfer learning model. Also, the authors of this dataset published a paper where they used an AI model trained by said data, perhaps if some of the parts of the transfer learning model were replaced by their model using new data that introduce variation will ultimately help build a very powerful model, an example of something that can be a focal point in the future aspect of this project.

## Limitations:
1) The proposed solution is highly overfit and incapable of generalizing well when introduced to new data. 
2) The proposed model is trained on hand gestures representing Arabic letters which is not accurate in terms of how deaf and mute people use sign language. 

## Future work: 
1) Train the model on images that have better resolution and overall quality.
2) To achieve real-time detection and deploy the model as a mobile application to reach a broader demographic. 
3) Train the model on video instead of images to expand its capabilities to interpret ASL more effectively.
4) Train the model on words to replicate how deaf and mute people actually use the sign language.


## Content 

```bash
./
│   README.md
|   .gitignore
|   Real time ASL Interpreter.pdf
|   ASL_Interpreter.ipynb
│   
└───model/
│   │   inference_graph/
|   |   ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/
|   |   ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config
|   |   ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz
│   
└───tfrecord_generation/
│   │   images/
|   |   generate_tfrecord.py
|   |   label_map.pbtxt
|   |   tfrecords_label_map.ipynb
│   
└───training/
    │   train/
    │   ckpt-xx.data
    |   ckpt-xx.index
    
```

## Used Libraries
`Tensorflow` 
`Object detection API` 
`CUDA`
`cuDNN`
