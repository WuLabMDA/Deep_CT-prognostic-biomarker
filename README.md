# Deep-CT-prognostic-biomarker
**Deep-CT predicts benefit from ICI therapy in NSCLC patients**  
The goal is to develop a deep learning-powered prognostic framework applied to baseline CT images to automate the quantification of patients’ risk of progression or death on ICI. Given the clinical observation that the metastatic patterns as manifested on CT can vary significantly across stage IV lung cancer patients, we focused on lung parenchymal and tumor lesions. To mitigate the uncertainty inevitable in one particular type of network model, we adopted an ensemble learning strategy to integrate fundamentally different but potentially complementary convolutional neural network (CNN) architectures to increase the model’s generalizability. The ensemble framework consisted of four 3D-CNN models (figure below), including a supervised learning network (sub-network 1), two hybrid networks that merge supervised and unsupervised learning differently (sub-networks 2, and 3), and an unsupervised learning network (sub-network 4).

![image](https://user-images.githubusercontent.com/77283272/214708177-c8b77f95-1855-47f2-a563-88315d8bcc91.png)

Please find the work in details here (later…)

**Data preparation**  
Deep-CT requires baseline CT from three different contrast as input; (1) lung window, (2) mediastinal window, and (3) The original/raw one. Figure below shows overall pipeline for data preparation. Note that, the final size of input down sampled to 128x128x128. 

![image](https://user-images.githubusercontent.com/77283272/214706511-6b2f1c80-cee7-4513-8ad7-2773e11263bc.png)

**Hardware requirements**  
Each sub-network was trained on NVIDIA A100 40GB/slot GPU unit, independently.

**Software requirements**  
Python, R, and Matlab were the platforms used for development. Sub-network 1 to 3 utilized Pytorch libraries (open source PyTorch v.1.4.0). Sub-network 4 mainly utilized R libraries (R 3.6.1). 

**Results**  

![image](https://user-images.githubusercontent.com/77283272/214708813-bcb25e6b-62e8-4e6b-950d-a29fdec72646.png)

**Code Structure and Explanation**  

