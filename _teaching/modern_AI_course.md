---
title: "Modern AI: Theory and application in biology"
collection: teaching
type: "Workshop"
permalink: /teaching/bioe-course-3
venue: "University of British Columbia"
date: 2026-7-01
location: "Vancouver, BC"
header:
  teaser: "teaching/modern_ai3.png"
---
This course introduces the core ideas behind modern AI, starting from the foundations of machine learning and neural networks and building toward today's most important model families. We will study how different data structures (tables, grids, sequences, sets, and graphs) motivate different architectures, how models are trained and generalized, and how modern generative, foundation, and multimodal systems are constructed. Along the way, we will connect these methods to applications in biological systems. Each lecture will be paired with practical notebooks and hands-on exercises so that students not only understand the concepts, but also learn how to implement and apply them in practice.



## Syllabus

## A. Foundations of Machine Learning and Neural Networks
**A.1** Foundations of machine learning: objectives, likelihoods, losses, supervised learning, and unsupervised learning  
**A.2** Gradient descent, backpropagation, gradient flow, and a practical introduction to PyTorch  

## B. Data Structures, Symmetries, and Neural Network Architectures
**B.1** Tabular data: MLPs, residual connections, and autoencoders  
**B.2** Grid-structured data: CNNs  
**B.3** Sequential data: RNNs, LSTMs, and GRUs  
**B.4** Graph-structured data: GNNs and message passing  
**B.5** Set-structured data: Deep Sets  
**B.6** Attention and Transformers
**B.7** A unifying view: invariance, equivariance, and geometric deep learning  

## C. Training, Inductive Bias, and Generalization
**C.1** Regularization, optimization, and practical training heuristics  
**C.2** Inductive bias and architecture choice  
**C.3** Generalization, out-of-distribution limits, and scaling laws  

## D. Generative Modeling
**D.1** A probabilistic view of data generation  
**D.2** Variational autoencoders  
**D.3** Flow matching models  
**D.4** Diffusion models  

## E. Learning Beyond Standard Supervision
**E.1** Transfer learning, fine-tuning, self-supervised learning, continual / online learning, and active learning  

## F. Multimodal Learning
**F.1** Why multimodal learning? Heterogeneity, complementarity, and shared representations  
**F.2** Combining modalities: fusion, joint embeddings, and cross-modal representation learning  
**F.3** Alignment and cross-modal interaction: correspondence, grounding, and attention  
**F.4** Multimodal reasoning and generation: from understanding to foundation models




<img width="799" height="482" alt="image" src="https://github.com/user-attachments/assets/70d8f726-101b-44ee-b37f-b25cca38de78" />




## Lectures & Notebooks

| # | Topic | Slides | Notebook |
|---|-------|--------|----------|
| 1 | **A.1** Foundations of machine learning: objectives, likelihoods, losses, supervised learning, and unsupervised learning | to be uploaded | to be uploaded |
| 2 | **A.2** Gradient descent, backpropagation, gradient flow, and a practical introduction to PyTorch | to be uploaded | to be uploaded |
| 3 | **B.1** Tabular data: MLPs, residual connections, and autoencoders | to be uploaded | to be uploaded |
| 4 | **B.2** Grid-structured data: CNNs | [PDF](https://raw.githubusercontent.com/TiamHeydari/tiamheydari.github.io/master/_teaching/Modern_AI/lectures/B.2.pdf) | [Notebook 1](https://colab.research.google.com/github/TiamHeydari/tiamheydari.github.io/blob/master/_teaching/Modern_AI/notebooks/Workshop_B.2_CNNs/0_helper_create_1D_image.ipynb), [Notebook 2](https://colab.research.google.com/github/TiamHeydari/tiamheydari.github.io/blob/master/_teaching/Modern_AI/notebooks/Workshop_B.2_CNNs/1_simple_one_layer_CNN_.ipynb), [Notebook 3](https://colab.research.google.com/github/TiamHeydari/tiamheydari.github.io/blob/master/_teaching/Modern_AI/notebooks/Workshop_B.2_CNNs/2_CNN_for_MedMNIST.ipynb) |
| 5 | **B.3** Sequential data: RNNs, LSTMs, and GRUs | [PDF](https://raw.githubusercontent.com/TiamHeydari/tiamheydari.github.io/master/_teaching/Modern_AI/lectures/B.3.pdf) | [Notebook 1](https://colab.research.google.com/github/TiamHeydari/tiamheydari.github.io/blob/master/_teaching/Modern_AI/notebooks/Workshop_B.3_RNNs/RNN_1_seq_to_one.ipynb), [Notebook 2](https://colab.research.google.com/github/TiamHeydari/tiamheydari.github.io/blob/master/_teaching/Modern_AI/notebooks/Workshop_B.3_RNNs/RNN_2_seq_to_seq_autoregressive.ipynb) |
| 6 | **B.4** Graph-structured data: GNNs and message passing | [PDF](https://raw.githubusercontent.com/TiamHeydari/tiamheydari.github.io/master/_teaching/Modern_AI/lectures/B.4.pdf) | [Notebook 1](https://colab.research.google.com/github/TiamHeydari/tiamheydari.github.io/blob/master/_teaching/Modern_AI/notebooks/Workshop_B.4_GNNs/1_CGNs_nodel_level.ipynb), [Notebook 2](https://colab.research.google.com/github/TiamHeydari/tiamheydari.github.io/blob/master/_teaching/Modern_AI/notebooks/Workshop_B.4_GNNs/2_CGNs_Graph_level.ipynb), [Notebook 3](https://colab.research.google.com/github/TiamHeydari/tiamheydari.github.io/blob/master/_teaching/Modern_AI/notebooks/Workshop_B.4_GNNs/3_MPNNs_nodel_level.ipynb), [Notebook 4](https://colab.research.google.com/github/TiamHeydari/tiamheydari.github.io/blob/master/_teaching/Modern_AI/notebooks/Workshop_B.4_GNNs/4_Todo_MPNNs_Graph_level.ipynb) |
| 7 | **B.5** Set-structured data: Deep Sets | [PDF](https://raw.githubusercontent.com/TiamHeydari/tiamheydari.github.io/master/_teaching/Modern_AI/lectures/B.5.pdf) | [Notebook 1](https://colab.research.google.com/github/TiamHeydari/tiamheydari.github.io/blob/master/_teaching/Modern_AI/notebooks/Workshop_B.5_Deep%20Sets/DeepSets1_supervised.ipynb), [Notebook 2](https://colab.research.google.com/github/TiamHeydari/tiamheydari.github.io/blob/master/_teaching/Modern_AI/notebooks/Workshop_B.5_Deep%20Sets/DeepSets2_AutoEncoder.ipynb) |
| 8 | **B.6** Attention and Transformers | [Part 1 PDF](https://raw.githubusercontent.com/TiamHeydari/tiamheydari.github.io/master/_teaching/Modern_AI/lectures/B.6%20Part%20I.pdf), [Part 2 PDF](https://raw.githubusercontent.com/TiamHeydari/tiamheydari.github.io/master/_teaching/Modern_AI/lectures/B.6%20Part%20II.pdf) | [Notebook 1](https://colab.research.google.com/github/TiamHeydari/tiamheydari.github.io/blob/master/_teaching/Modern_AI/notebooks/Workshop_B.6_Attention%20and%20Transformers/1_RNN_seq_to_seq_with_attention.ipynb) |
| 9 | **B.7** A unifying view: invariance, equivariance, and geometric deep learning | to be uploaded | to be uploaded |
| 10 | **C.1** Regularization, optimization, and practical training heuristics | to be uploaded | to be uploaded |
| 11 | **C.2** Inductive bias and architecture choice | to be uploaded | to be uploaded |
| 12 | **C.3** Generalization, out-of-distribution limits, and scaling laws | to be uploaded | to be uploaded |
| 13 | **D.1** A probabilistic view of data generation | to be uploaded | to be uploaded |
| 14 | **D.2** Variational autoencoders | to be uploaded | to be uploaded |
| 15 | **D.3** Flow matching models | to be uploaded | to be uploaded |
| 16 | **D.4** Diffusion models | to be uploaded | to be uploaded |
| 17 | **E.1** Transfer learning, fine-tuning, self-supervised learning, continual / online learning, and active learning | to be uploaded | to be uploaded |
| 18 | **F.1** Why multimodal learning? Heterogeneity, complementarity, and shared representations | to be uploaded | to be uploaded |
| 19 | **F.2** Combining modalities: fusion, joint embeddings, and cross-modal representation learning | to be uploaded | to be uploaded |
| 20 | **F.3** Alignment and cross-modal interaction: correspondence, grounding, and attention | to be uploaded | to be uploaded |
| 21 | **F.4** Multimodal reasoning and generation: from understanding to foundation models | to be uploaded | to be uploaded |


> **Note:** Upload PDF slides to `files/teaching/modern-ai/` using the filenames above (e.g. `lecture-01.pdf`) and they will automatically become active links. Replace each `[Colab](#)` placeholder with the corresponding Google Colab notebook URL.
