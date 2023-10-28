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

#### Related Papers

- [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)
- [Sky pixel detection in outdoor imagery using an adaptive algorithm
and machine learning](https://arxiv.org/abs/1910.03182)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
- [A Hybrid Thresholding Algorithm for Cloud Detection on Ground-Based Color Images](https://journals.ametsoc.org/view/journals/atot/28/10/jtech-d-11-00009_1.xml)
- [Semantic Understanding of Scenes through the ADE20K Dataset](https://arxiv.org/abs/1608.05442)
- [SUN Database: Large-scale Scene Recognition from Abbey to Zoo](https://vision.princeton.edu/projects/2010/SUN/paper.pdf)
- [The Mapillary Vistas Dataset for Semantic Understanding of Street Scenes](https://openaccess.thecvf.com/content_ICCV_2017/papers/Neuhold_The_Mapillary_Vistas_ICCV_2017_paper.pdf)
- [The Cityscapes Dataset for Semantic Urban Scene Understanding](https://arxiv.org/abs/1604.01685)
- [Sky segmentation in the wild: An empirical study](https://ieeexplore.ieee.org/document/7477637)
- [Color-based Segmentation of Sky/Cloud Images
From Ground-based Cameras](https://stefan.winkler.site/Publications/jstars2017.pdf)
- [AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients](https://arxiv.org/abs/2010.07468)

### Week 03

#### Objectives

- [x] Create segmentation dataset for sky/cloud segmentation.
- [x] Work on sky/cloud segmentation.

#### Progress

This week, I dived into the sky/cloud segmentation process. The initial approach involved utilizing superpixels and a decision tree classifier for segmenting and classifying various sections of the sky. Each superpixel in an image was analyzed based on its descriptor vector.

#### Challenges and Solutions

While this approach was generally effective, it encountered limitations in distinguishing complex cloud formations and varied lighting conditions. In certain scenarios, the algorithm misclassified cloud features, underscoring the need for refinement. I decided to use deep learning to improve the performance of the algorithm in the next week.

#### Related Papers

- [SLIC Superpixels Compared to State-of-the-Art Superpixel Methods](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6205760)

### Week 04

#### Objectives

- [x] Work on sky/cloud segmentation using deep learning.

#### Progress

This week, I implemented a deep learning model for sky/cloud segmentation. I used the DeepLabV3 architecture with a ResNet-101 backbone. The model was trained on the previously made custom dataset. The model was able to accurately segment the sky and clouds in the test images.

#### Challenges and Solutions

Overall, the model performed well, but it was unable to accurately segment the sky in certain scenarios. This was due to the lack of sufficient training data. In the upcoming weeks, I will focus on detecting changes in light within the sky and utilize optical flow to track motion.

#### Related Papers

- [Automatic Cloud Detection for All-Sky Images
Using Superpixel Segmentation](https://ieeexplore.ieee.org/document/6874559)
- [Multi-level semantic labeling of Sky/cloud images](https://ieeexplore.ieee.org/document/7350876)

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
Additionally, I applied optical flow on images that are still warped, which affects the accuracy and reliability of the data derived from these images. To address this issue, it is essential to retrieve and utilize camera intrinsic parameters. By unwarping the images, the accuracy of data obtained through the optical flow algorithm will be improved.

#### Related Papers

- [Detection of changes in luminance distributions](https://jov.arvojournals.org/article.aspx?articleid=2121050)

### Week 06

#### Objectives

- [x] Extend the global lighting analysis across various scenes.
- [ ] Retrieve camera intrinsic parameters.
- [x] Implement image unwarping to rectify the images before analyzing them.
- [x] Incorporate the RAFT deep learning-based optical flow algorithm.

#### Progress

The **Global Lighting Evaluation** was expanded this week, incorporating both boxplot and violin plot methodologies to discern distribution disparities across different scenes. The **RAFT** deep learning optical flow algorithm was introduced to the analysis, aiming to offer a more nuanced understanding of scene dynamics. There was a marked difference observed in the optical flow magnitudes between clear sky and overcast images when using RAFT. The clearer skies exhibited more diverse optical flow magnitudes, possibly due to a scarcity of features in overcast images to anchor the flow.

**Image unwarping** was also tackled, aimed at ameliorating angle accuracy in the context of optical flow. This effort was primarily manual, given the absence of the camera's intrinsic parameters.

#### Challenges and Solutions

A comparison between the Farneback and RAFT algorithms unveiled distinct differences. While Farneback appeared to hone in on cloud peripheries, RAFT seemed to center its attention on the clouds themselves. However, in certain scenes, RAFT's outputs did not seem to align with the actual L1 differences between frames. This suggests that RAFT might be zeroing in on distinct scene elements than initially anticipated.

The manual unwarping process, while helpful, wasn't flawless. To further enhance the rectification process, considering alternative projections, such as equirectangular, might prove beneficial. As we move forward, it would be prudent to conduct a deeper dive to determine whether RAFT is the optimal metric for lighting change detection or if the Farneback method offers superior insights.

### Week 07

#### Objectives

- [ ] Check segmentation models on Dong's HDR video scenes.
- [ ] Extend last week's optical flow analysis using Farneback's algorithm.
- [ ] Verify optical flow with bidirectional comparison and bilinear sampling.
- [ ] Use different projections to keep distances consistent instead of angles.

#### Progress

TODO

#### Challenges and Solutions

TODO

### Other Tasks

- [?] Focus on ground.
- [?] Integrate shadow detection to better understand lighting changes.
- [?] Research and implement shadow detection algorithms.
- [?] Analyze how shadows correlate with other observed lighting changes.
- [?] Develop a real-time processing pipeline for video analysis.
- [?] Develop a user interface to visualize the analysis results.

_Note: This README is a template and will be updated as the project progresses._
