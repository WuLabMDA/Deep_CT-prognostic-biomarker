# Deep-CT-prognostic-biomarker
**Deep-CT predicts benefit from ICI therapy in NSCLC patients**  
The goal is to develop a deep learning-powered prognostic framework applied to baseline CT images to automate the quantification of patients’ risk of progression or death on ICI. Given the clinical observation that the metastatic patterns as manifested on CT can vary significantly across stage IV lung cancer patients, we focused on lung parenchymal and tumor lesions. To mitigate the uncertainty inevitable in one particular type of network model, we adopted an ensemble learning strategy to integrate fundamentally different but potentially complementary convolutional neural network (CNN) architectures to increase the model’s generalizability. The ensemble framework consisted of four 3D-CNN models (figure below), including a supervised learning network (sub-network 1), two hybrid networks that merge supervised and unsupervised learning differently (sub-networks 2, and 3), and an unsupervised learning network (sub-network 4).

![image](https://github.com/WuLabMDA/Deep_CT-prognostic-biomarker/assets/77283272/568ad6a8-f2b9-4dd3-8b3c-64b48f12d37f)


Please find the work in details here https://authors.elsevier.com/sd/article/S2589-7500(23)00082-1

**Data preparation**  
Deep-CT requires baseline CT from three different contrast as input; (1) lung window, (2) mediastinal window, and (3) The original/raw one. Figure below shows overall pipeline for data preparation. Note that, the final size of input was down sampled to 128x128x128. 

![image](https://user-images.githubusercontent.com/77283272/214706511-6b2f1c80-cee7-4513-8ad7-2773e11263bc.png)

**Hardware requirements**  
Each sub-network was trained on NVIDIA A100 40GB/slot GPU unit, independently.

**Software requirements**  
Python, R, and Matlab were the platforms used for development. Sub-network 1 to 3 utilized Pytorch libraries (open source PyTorch v.1.4.0). Sub-network 4 mainly utilized R libraries (R 3.6.1). 

**Results**  
![Results](https://github.com/WuLabMDA/Deep_CT-prognostic-biomarker/assets/77283272/2b8a69b1-b736-4705-ba66-761c7efb0144)



**Code Structure and Explanation**  
1. Codes are uploaded in terms of the sub-components of Deep-CT (sub-network1 to sub-network4 foldeers),and how the individual sub-componet was later ensembled as Deep-CT.
2. We are unable to provide the raw data due to privacy concerns, however we did include dummy data in an effort to demonstrate how the codes run.
3. Please run each sub-networks independently to get their respective independent risk (subgroups). You will have four risks in total correspoding to four sub-network.
4. Once you have the final four sub-network ouput, please combined them into one spreadsheet (four columns) and feed it codes in Deep-CT folder. The ensemble of the four sub-networks is done via Random Survival Forest.

