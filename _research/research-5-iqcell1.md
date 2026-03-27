---
title: "iQcell1: framing gene regulatory networks from scRNAseq data as a Boolean satisfiability problem (SAT)"
collection: research
permalink: /research/iqcell1
excerpt: 'Description will be added here.'
header:
  teaser: "research/iqc1.png"
---

IQCELL: A platform for predicting the effect of gene perturbations on developmental trajectories using single-cell RNA-seq data introduced the first version of our effort to reconstruct executable gene regulatory networks directly from single-cell transcriptomic data. The central idea was that scRNA-seq should not only be used to describe cell states, but also to build dynamic models that can be simulated, perturbed, and tested as hypotheses about developmental control. IQCELL combined gene selection, dropout correction, pseudotime-aware network inference, logical rule construction, and dynamical simulation into a single framework for studying how transcription factors coordinate developmental trajectories.

<img width="512" height="474" alt="image" src="https://github.com/user-attachments/assets/3fbff6b2-8057-4e0c-b3b7-f7386c797acf" />



Applied to early mouse T-cell and erythroid development, it recovered over 3/4 of previously reported causal interactions and qualitatively reproduced known perturbation effects, showing that single-cell data could be used not only to describe cell states but also to build predictive, executable models of developmental control.

Later, this work served as a foundation for later studies in human developmental systems. In a subsequent study of human mast, myeloid, and T-lineage specification, gene-regulatory-network inference within this broader framework uncovered a regulatory role for YBX1 in human T-lineage specification, and experimental perturbation showed that loss of YBX1 reduced T-lineage output. This helped extend the original vision of IQCELL from proof-of-principle network inference toward actual biological discovery in human systems.


<img width="1842" height="712" alt="image" src="https://github.com/user-attachments/assets/9f8ba6b4-6df1-4d82-9074-705a74d28afc" />


Read more: <br>
https://gitlab.com/stemcellbioengineering/iqcell <br>
https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009907 <br>
https://www.cell.com/cell-systems/fulltext/S2405-4712(24)00310-7

