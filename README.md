# Deep-CT-prognostic-biomarker
**Deep-CT predicts benefit from ICI therapy in NSCLC patients**  
The goal is to predict benefit from immune checkpoint inhibitor (ICI) therapy in EGFR/ALK wild-type NSCLC, in large multi-institution cohorts (n=976). Deep-CT is an ensemble deep learning signature developed and externally validated for the accurate prediction of progression-free survival and overall survival from baseline CT images, independent of known clinicopathological variables including PD-L1.   
Please find the work in details here (later…)

**Data preparation**  
Deep-CT requires baseline CT from three different contrast as input; (1) lung window, (2) mediastinal window, and (3) The original/raw one. Figure below shows overall pipeline for data preparation. Note that, the final size of input down sampled to 128x128x128. 

![image](https://user-images.githubusercontent.com/77283272/214706511-6b2f1c80-cee7-4513-8ad7-2773e11263bc.png)

**Hardware requirements**  
Each sub-network was trained on NVIDIA 40GB xxxx GPU unit, independently.

**Software requirements**  
Python, R, and Matlab were the platforms used for development. Sub-network 1 to 3 utilized Pytorch libraries. Sub-network 4 mainly utilized R libraries. 
