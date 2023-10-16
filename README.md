# 🌞 Analyzing the Changes in Captured Daylight in Real-Time Videos

## 📋 Project Overview

This project aims to develop new methods for analyzing changes in daylight conditions captured in real-time videos. We focus on detecting, extracting, and tracking temporal changes, motion, and luminous changes in video footage.

## 📖 Log Book

### Week 01

#### Objectives

- [x] Create a GitHub repository for the project.
- [x] Begin the project with an introductory overview.

#### Progress

I created a GitHub repository for the project and I read some papers about how to segment sky images using basic image processing techniques.

#### Challenges and Solutions

I recognized the need to accurately detect and exclude the ground before classifying the sky to ensure more precise analysis. My upcoming tasks will focus on video preprocessing and efficient ground detection.

### Week 02

#### Objectives

- [x] Work on video preprocessing.
- [x] Work on ground detection.

#### Progress

I applied Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance cloud features but noted the introduction of noise. I mitigated this with Bilateral Filtering and temporal averaging. For ground detection, traditional image processing techniques were insufficient, so I implemented a deep learning model which proved more effective.

#### Challenges and Solutions

Basic techniques like thresholding, clustering, superpixels, and edge detection were limited and did not offer optimal performance for ground detection. I decided to explore deep learning approaches, which proved to be more effective in accurately identifying and isolating ground features.

### Week 03

#### Objectives

- [x] Create segmentation dataset for sky/cloud segmentation.
- [x] Work on sky/cloud segmentation.

#### Progress

This week, I dived into the sky/cloud segmentation process. The initial approach involved utilizing superpixels and a decision tree classifier for segmenting and classifying various sections of the sky. Each superpixel in an image was analyzed based on its descriptor vector.

#### Challenges and Solutions

While this approach was generally effective, it encountered limitations in distinguishing complex cloud formations and varied lighting conditions. In certain scenarios, the algorithm misclassified cloud features, underscoring the need for refinement. I decided to use deep learning to improve the performance of the algorithm in the next week.

### Week 04

#### Objectives

- [x] Work on sky/cloud segmentation using deep learning.

#### Progress

This week, I implemented a deep learning model for sky/cloud segmentation. I used the DeepLabV3 architecture with a ResNet-101 backbone. The model was trained on the previously made custom dataset. The model was able to accurately segment the sky and clouds in the test images.

#### Challenges and Solutions

Overall, the model performed well, but it was unable to accurately segment the sky in certain scenarios. This was due to the lack of sufficient training data. In the upcoming weeks, I will focus on detecting changes in light within the sky and utilize optical flow to track motion.

## ⌨️ Commands
```
pip install -e .
```

_Note: This README is a template and will be updated as the project progresses._
