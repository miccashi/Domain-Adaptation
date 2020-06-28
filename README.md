# DomainAdaptation

1 Introduction

When using machine learning techniques to tackle real-world problems, training robust models for eachproblemisnotrealistic. Learning from heterogeneous domains and generalizing existing models to perform well on target domains are important. This project concentrates on conducting a domain adaptation experiment where we mimic the transfer learning scenario with only a few target domain labels available. And feature-representation transfer learning as one case of feasible approaches intuited by that the knowledge in features can be learned and transferred to target domains, can be used to tackle the problem (Pan & Yang, 2009). In this project, we implement the feature augmentation technique (Daumé III, 2009) and observe the performance on the dataset of students’ examination records at the U.K. Additionally, we conduct an extension approach - a semi-supervised method utilizing unseen in-domain data to perform domain adaptation problem (Daumé III et al., 2010).

2 Dataset

We perform three tasks on the dataset and name tasks as target-domain-tasks. Also, We use 70%, 15%, and 15% of each dataset as train, validation, and test data out of 4404, 3654, and 7304 instances for male-only, female-only and mixed-gender schools, respectively. Any domain as the target case would have only 100 labelled examples for training and evaluating. Additionally, one-hot encoding is used on categorical data.

