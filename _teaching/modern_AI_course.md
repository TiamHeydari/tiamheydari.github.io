---
title: "Modern AI: Theory and application in biology"
collection: teaching
type: "Workshop"
permalink: /teaching/bioe-course-3
date: 2026-07-01
venue: "University of British Columbia"
header:
  teaser: "teaching/modern_ai5.png"
---

<img width="1024" height="1024" alt="modern_ai5" src="https://github.com/user-attachments/assets/fea3d216-0e81-4b60-9533-24357e677a03" />


## This workshop is under development

This course introduces the core ideas behind modern AI, starting from the foundations and building toward today's most important approaches. We will study how different data structures (tables, grids, sequences, sets, and graphs) motivate different architectures, how models are trained and generalized, and how modern generative, foundation, and multimodal systems are constructed. Along the way, we will connect these methods to applications in biological systems. Most lectures will be paired with practical notebooks and hands-on exercises so that students not only understand the concepts, but also learn how to implement and apply them in practice.



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

## C. Attention and Transformers
**C.1** Origins of attention  
**C.2** Attention as a standalone operator  
**C.3** Transformer architectures  
**C.4** Transformers across different data structures  
**C.5** Training transformers  

## D. Representation Learning and Generative Modeling
**D.1** Representation Learning: Learning useful coordinates for data  
**D.2** Representation Learning: Evaluating and probing representations  
**D.3** Representation Learning: How to learn a good representation  
**D.4** Generative modeling: Variational autoencoders  
**D.5** Generative modeling: Flow matching models  
**D.6** Generative modeling: Diffusion models  

## E. Geometric Deep Learning
**E.1** A unifying view: invariance, equivariance, and geometric deep learning  

## F. Multimodal Learning
**F.1** Why multimodal learning? Heterogeneity, complementarity, and shared representations  
**F.2** Combining modalities: fusion, joint embeddings, and cross-modal representation learning  
**F.3** Alignment and cross-modal interaction: correspondence, grounding, and attention  
**F.4** Multimodal reasoning and generation: from understanding to foundation models


## Lectures & Notebooks

| # | Topic | Slides | Notebook |
|---|-------|--------|----------|
| 1 | **A.1** Foundations of machine learning: objectives, likelihoods, losses, supervised learning, and unsupervised learning | to be uploaded | to be uploaded |
| 2 | **A.2** Gradient descent, backpropagation, gradient flow, and a practical introduction to PyTorch | to be uploaded | to be uploaded |
| 3 | **B.1** Tabular data: MLPs, residual connections, and autoencoders | to be uploaded | to be uploaded |
| 4 | **B.2** Grid-structured data: CNNs | <a href="https://raw.githubusercontent.com/TiamHeydari/tiamheydari.github.io/master/_teaching/Modern_AI/lectures/B.2.pdf" target="_blank" rel="noopener">PDF</a> | [Notebook 1](https://colab.research.google.com/github/TiamHeydari/tiamheydari.github.io/blob/master/_teaching/Modern_AI/notebooks/Workshop_B.2_CNNs/0_helper_create_1D_image.ipynb), [Notebook 2](https://colab.research.google.com/github/TiamHeydari/tiamheydari.github.io/blob/master/_teaching/Modern_AI/notebooks/Workshop_B.2_CNNs/1_simple_one_layer_CNN_.ipynb), [Notebook 3](https://colab.research.google.com/github/TiamHeydari/tiamheydari.github.io/blob/master/_teaching/Modern_AI/notebooks/Workshop_B.2_CNNs/2_CNN_for_MedMNIST.ipynb) |
| 5 | **B.3** Sequential data: RNNs, LSTMs, and GRUs | <a href="https://raw.githubusercontent.com/TiamHeydari/tiamheydari.github.io/master/_teaching/Modern_AI/lectures/B.3.pdf" target="_blank" rel="noopener">PDF</a> | [Notebook 1](https://colab.research.google.com/github/TiamHeydari/tiamheydari.github.io/blob/master/_teaching/Modern_AI/notebooks/Workshop_B.3_RNNs/RNN_1_seq_to_one.ipynb), [Notebook 2](https://colab.research.google.com/github/TiamHeydari/tiamheydari.github.io/blob/master/_teaching/Modern_AI/notebooks/Workshop_B.3_RNNs/RNN_2_seq_to_seq_autoregressive.ipynb) |
| 6 | **B.4** Graph-structured data: GNNs and message passing | <a href="https://raw.githubusercontent.com/TiamHeydari/tiamheydari.github.io/master/_teaching/Modern_AI/lectures/B.4.pdf" target="_blank" rel="noopener">PDF</a> | [Notebook 1](https://colab.research.google.com/github/TiamHeydari/tiamheydari.github.io/blob/master/_teaching/Modern_AI/notebooks/Workshop_B.4_GNNs/1_CGNs_nodel_level.ipynb), [Notebook 2](https://colab.research.google.com/github/TiamHeydari/tiamheydari.github.io/blob/master/_teaching/Modern_AI/notebooks/Workshop_B.4_GNNs/2_CGNs_Graph_level.ipynb), [Notebook 3](https://colab.research.google.com/github/TiamHeydari/tiamheydari.github.io/blob/master/_teaching/Modern_AI/notebooks/Workshop_B.4_GNNs/3_MPNNs_nodel_level.ipynb), [Notebook 4](https://colab.research.google.com/github/TiamHeydari/tiamheydari.github.io/blob/master/_teaching/Modern_AI/notebooks/Workshop_B.4_GNNs/4_Todo_MPNNs_Graph_level.ipynb) |
| 7 | **B.5** Set-structured data: Deep Sets | <a href="https://raw.githubusercontent.com/TiamHeydari/tiamheydari.github.io/master/_teaching/Modern_AI/lectures/B.5.pdf" target="_blank" rel="noopener">PDF</a> | [Notebook 1](https://colab.research.google.com/github/TiamHeydari/tiamheydari.github.io/blob/master/_teaching/Modern_AI/notebooks/Workshop_B.5_Deep%20Sets/DeepSets1_supervised.ipynb), [Notebook 2](https://colab.research.google.com/github/TiamHeydari/tiamheydari.github.io/blob/master/_teaching/Modern_AI/notebooks/Workshop_B.5_Deep%20Sets/DeepSets2_AutoEncoder.ipynb) |
| 8 | **C.1** Origins of attention | <a href="https://raw.githubusercontent.com/TiamHeydari/tiamheydari.github.io/master/_teaching/Modern_AI/lectures/C.1.pdf" target="_blank" rel="noopener">PDF</a> | [Notebook 1](https://colab.research.google.com/github/TiamHeydari/tiamheydari.github.io/blob/master/_teaching/Modern_AI/notebooks/Workshop_B.6_Attention%20and%20Transformers/1_RNN_seq_to_seq_with_attention.ipynb) |
| 9 | **C.2** Attention as a standalone operator | <a href="https://raw.githubusercontent.com/TiamHeydari/tiamheydari.github.io/master/_teaching/Modern_AI/lectures/C.2.pdf" target="_blank" rel="noopener">PDF</a> | to be uploaded |
| 10 | **C.3** Transformer architectures | <a href="https://raw.githubusercontent.com/TiamHeydari/tiamheydari.github.io/master/_teaching/Modern_AI/lectures/C.3.pdf" target="_blank" rel="noopener">PDF</a> | to be uploaded |
| 11 | **C.4** Transformers across different data structures | <a href="https://raw.githubusercontent.com/TiamHeydari/tiamheydari.github.io/master/_teaching/Modern_AI/lectures/C.4.pdf" target="_blank" rel="noopener">PDF</a> | to be uploaded |
| 12 | **C.5** Training transformers | to be uploaded | to be uploaded |
| 13 | **D.1** Representation Learning: Learning useful coordinates for data | <a href="https://raw.githubusercontent.com/TiamHeydari/tiamheydari.github.io/master/_teaching/Modern_AI/lectures/D.1.pdf" target="_blank" rel="noopener">PDF</a> | [Notebook 1](https://colab.research.google.com/github/TiamHeydari/tiamheydari.github.io/blob/master/_teaching/Modern_AI/notebooks/Workshop_D.1_Representation_Learning/1_functions_as_mappings.ipynb), [Notebook 2](https://colab.research.google.com/github/TiamHeydari/tiamheydari.github.io/blob/master/_teaching/Modern_AI/notebooks/Workshop_D.1_Representation_Learning/2_simple_MLP_as_mapping.ipynb), [Notebook 3](https://colab.research.google.com/github/TiamHeydari/tiamheydari.github.io/blob/master/_teaching/Modern_AI/notebooks/Workshop_D.1_Representation_Learning/3_Manifold_hypothesis_of_CNN_for_MedMNIST.ipynb), [Notebook 4](https://colab.research.google.com/github/TiamHeydari/tiamheydari.github.io/blob/master/_teaching/Modern_AI/notebooks/Workshop_D.1_Representation_Learning/4_Manifold_hypothesis_of_CNN_for_MedMNIST_better_statistics.ipynb) |
| 14 | **D.2** Representation Learning: Evaluating and probing representations | to be uploaded | to be uploaded |
| 15 | **D.3** Representation Learning: How to learn a good representation | to be uploaded | to be uploaded |
| 16 | **D.4** Generative modeling: Variational autoencoders | to be uploaded | to be uploaded |
| 17 | **D.5** Generative modeling: Flow matching models | to be uploaded | to be uploaded |
| 18 | **D.6** Generative modeling: Diffusion models | to be uploaded | to be uploaded |
| 19 | **E.1** A unifying view: invariance, equivariance, and geometric deep learning | to be uploaded | to be uploaded |
| 20 | **F.1** Why multimodal learning? Heterogeneity, complementarity, and shared representations | to be uploaded | to be uploaded |
| 21 | **F.2** Combining modalities: fusion, joint embeddings, and cross-modal representation learning | to be uploaded | to be uploaded |
| 22 | **F.3** Alignment and cross-modal interaction: correspondence, grounding, and attention | to be uploaded | to be uploaded |
| 23 | **F.4** Multimodal reasoning and generation: from understanding to foundation models | to be uploaded | to be uploaded |


> **Note:** Slide links above use raw GitHub URLs and are set to open in a new browser tab. Notebook links point to Google Colab versions.
