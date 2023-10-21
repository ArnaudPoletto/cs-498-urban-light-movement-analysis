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

### Week 05

#### Objectives

- [x] Work on lighting analysis from consecutive video frames.

#### Progress

This week focused heavily on conducting an in-depth analysis of lighting variations captured in consecutive video frames. The primary objective was to develop a robust methodology to quantitatively measure and characterize these lighting changes, enabling the identification and analysis of patterns and anomalies.

**Global Lighting Evaluation** was conducted, utilizing several statistical evaluations. By computing the properties between video frames at 3-second intervals, I was able to observe both abrupt and gradual changes in lighting. Four methods were employed:

- **L1 Difference**: Measuring the absolute differences in pixel values between consecutive frames helped identify abrupt changes in lighting.
- **Mean Value Difference**: This metric was instrumental in tracking gradual changes in lighting by computing the average pixel value differences between frames.
- **Brightness Offset Detection**: Utilized histogram peak detection to identify shifts in the overall brightness of the scene.
- **Earth Mover’s Distance**: This method was used to capture the dissimilarity between two sequential frame histograms, offering insights into the global changes in brightness and color distribution.

The **Dual-Zone Lighting Evaluation** was another focal point. I applied Otsu’s method on the L channel from LAB to segment the sky into light and dark zones. This segmentation was conducted at different granularities, including superpixel and pixel levels. Four statistical evaluations were applied:
- **Percentage of Light Pixels**: This quantified the proportion of lighter pixels in the frames, offering insights into the overall brightness.
- **Signed Difference in Percentage of Light Pixels**: This helped in identifying the variations in the proportion of light pixels between consecutive frames.
- **L1 Difference between Lighting Masks**: This measurement offered insights into the abrupt changes in lighting conditions.
- **Mean Value Difference for Light/Dark Areas**: This metric aided in assessing the gradual variations in both light and dark areas.

Lastly, the **Optical Flow's Farneback Algorithm** was employed to obtain optical flow vectors from consecutive frames using the L channel of the LAB color space. This allowed for a detailed analysis of the movement and changes in lighting, contributing to a comprehensive understanding of the dynamic lighting conditions.

#### Challenges and Solutions

The computation of statistical properties between video frames at 3-second intervals presented challenges in terms of processing time and data management. The segmentation of the sky into light and dark zones was initially inconsistent, leading to unreliable data. Additional efforts and refinements are required to optimize the computational efficiency and enhance the consistency of sky segmentation if these methodologies prove to be valuable in our ongoing lighting analysis.

## ⌨️ Commands
```
pip install -e .
```

_Note: This README is a template and will be updated as the project progresses._
