# TDM Project Closure Document — Instructions & Examples

> This document serves as the official record for the project, summarizing the work completed, key outcomes, challenges encountered, and lessons learned. It also confirms that project materials have been organized and shared with mentors to support future continuation of the work if needed.
>
> Teams should use the [provided template](https://the-examples-book.com/crp/students/_attachments/TDM%20Project%20Closure.dotx) to complete this document. **A complete response often requires at least three sentences per section.**

---

## Section 1: Project Overview

This project, developed in partnership with **The Aerospace Corporation**, explores the application of machine learning to automate the detection and classification of radio frequency (RF) interference signals. As wireless communication technologies expand, RF interference poses increasingly disruptive challenges, particularly for systems reliant on accurate satellite-based navigation like GPS and GNSS constellations. Monitoring RF environments manually is difficult due to the volume and complexity of the data, which motivated our approach to use deep learning pipelines.

| Field | Description |
|---|---|
| **Project Name** | RF Interference |
| **Corporate Partner** | The Aerospace Corporation |
| **Student Team (Development Team)** | Eubene In, Ashmit Tendolkar, Vineet Rao, Rex Wu, Ronak Mohanty, Adarsh Rangayyan, Tushar Bhat, Sreya Etherajan  |
| **Teaching Assistant (Scrum Master)** | William Yu |
| **Corporate Partner Mentor(s) (Product Owner(s))** | Kyle Logue, Anson Lim |
| **Project Semester(s)** | Spring 2026 |

---

## Section 2: Executive Summary

A concise overview of the project and its final outcomes. Teams should review what was accomplished, how the scope evolved, and what value the work provides to corporate partners.

### 2.1 Objective

The objective of this project was to understand GNSS jamming and automate the detection and classification of radio frequency (RF) interference signals using machine learning. The goal was to classify over 20+ GB of noisy satellite data from the Highway2 GNSS Jamming Dataset into one of 9 predetermined classes.

---

### 2.2 Scope

The project involved treating RF signal classification as a multi-class image classification problem. The team processed Power Spectral Density (PSD) matrices through a preprocessing pipeline (normalization, downsampling, and resizing to 224x224). Instead of a single model, we trained multiple architectures—including Convolutional Neural Networks (CNNs), Vision Transformers (ViT), and Audio Spectrogram Transformers (AST) using PyTorch Lightning—and aggregated them into an ensemble using majority voting.

---

### 2.3 Outcome

We developed multiple distinct ML pipelines and successfully constructed an optimized 11-model ensemble. The models learned to independently evaluate processed PSD matrices to output a predicted interference class. An earlier 11-model configuration reached about **75.42%** test accuracy; after further tuning, the ensemble reached **76.91%** on the test set.

---

### 2.4 Business Impact

Manually monitoring RF environments is difficult due to the sheer volume and complexity of the signal data. This project demonstrates that machine learning is a promising, scalable approach for automating this detection. Ultimately, it lays the groundwork for systems that support more automated RF monitoring in real-world environments, protecting satellite-based navigation systems (like GPS/GNSS) against safety risks from jamming.



---

## Section 3: Key Achievements

Documents the final outputs produced by the team. Focus on the **most important results**, not every task completed.

### 3.1 Final Deliverables

- Model training scripts
- Visualization tools
- Code repositories, scripts and notebooks
- Technical documentation or setup guides
- No model weights since they are too large for github

---

### 3.2 Code Repository

This repo contains all of our source code and model training scripts

> `[https://github.com/ubean-nn/TDM-AerospaceCorp-RF-Interference](https://github.com/ubean-nn/TDM-AerospaceCorp-RF-Interference)`

---

### 3.3 Data & Reports Location

> All datasets can be found here: `https://gitlab.cc-asp.fraunhofer.de/darcy_gnss/fiot_highway2`

---

### 3.4 Documentation

Technical documentation is in the READMEs under `workspaces/`; a short model overview is in `workspaces/model-arch.md`.

---

### 3.5 Achievements

We built an 11-model ensemble (majority voting) and improved test accuracy from about **75.42%** to **76.91%** after optimization; individual member models spanned a wide range of performance, which motivated combining them. We reported **accuracy and F1** (and related evaluation) on the 9-class task, and we delivered end-to-end training from PSD / spectrogram inputs through **PyTorch Lightning**, using the project’s data tooling (e.g. `Highway2Dataset` and the shared workspace layout).

---

## Section 4: Challenges and Lessons Learned (Project Reflection)

Reflects on the project experience and documents insights for future teams. Focus on key experiences — do **not** repeat deliverables already listed above.

### 4.1 Challenges Encountered

The primary challenges were the sheer volume and complexity of the raw RF spectrum data and the severe imbalance in the Highway2 GNSS Jamming Dataset.

- **Class imbalance** made single metrics and per-class performance uneven across the nine labels.
- **Model variance:** individual architectures and training runs differed widely in accuracy, complicating reliance on any one model.
- **Preprocessing sensitivity:** choices in normalization, downsampling, and resizing had an outsized effect on results given the noise and size of the data.

---

### 4.2 Mitigation Strategies or Resolutions

To address the data imbalance, we utilized SMOTE to generate fake samples for minority classes and implemented Weighted/Focal Loss during training. To address variations in model reliability, rather than relying on a single architecture, we employed an ensemble approach with majority voting, which made weaker models significantly stronger when aggregated. Computation speed was managed by training models using GPU resources which we found to be fastest at night when demand was low and the queue was short.

---

### 4.3 Lessons Learned

One major lesson we learned was that data preprocessing is just as important as the model's architecture itself. Since the RF interference data was large, noisy, and imbalanced, the way we normalized, downsampled, resized, and organized the PSD matrices had a major impact on model performance. As a result, we highly recommend future teams to take significant time to understand the data as much as possible before diving head first into making a model. Additionally, we also learned that every model architecture has its own strengths and weaknesses. Thus, we found that an ensemble approach would not only be more holistic but also more representative of each one of ours' work. 

---

## Section 5: Next Steps & Project Closure

Summarizes how the project is concluded and how results may be continued in the future. Confirms that materials are organized for mentor handover.

### 5.1 Handover Summary

We shared the GitHub repository, this closure document, and the `workspaces/` documentation (including per-area READMEs) for mentor and partner follow-up. A project poster summarized methodology, individual model results, and ensemble metrics for presentations to stakeholders and The Data Mine.

---

### 5.2 Project Closure Checklist

Confirm that key materials have been prepared and shared. Mark all items that apply:

- [ ] Creation of a Project Closure document
- [ ] Code repository shared with mentors
- [ ] Project documentation provided (README, setup instructions, architecture overview)
- [ ] Data sources or datasets documented (if applicable)
- [ ] Environment setup or dependencies documented (if applicable)
- [ ] Known issues or limitations documented
- [ ] Recommendations or next steps for future work documented
- [ ] Final project walkthrough or presentation completed with mentors

---

### 5.3 Next Steps & Recommendations

**What to write:** Provide suggestions for how the project could be improved or extended in the future. If recommendations are already in your project documentation, summarize them here or reference their location. Examples include:

- Improving model performance
- Expanding the dataset
- Scaling the solution for production use
- Integrating additional data sources or features

---

### 5.4 Sign-Off (Formal Acceptance)

This section confirms that project results and deliverables have been reviewed by the corporate partner or mentor, representing formal acknowledgement that the project work has been completed and shared.

| Role | Name | Date | Signature |
|---|---|---|---|
| Corporate Partner Mentor | | | |
| Teaching Assistant | | | |
| Team Lead / Representative | | | |

---

### 5.5 Additional Notes *(Optional)*

Any other important information that does not fall under the sections above.

---
