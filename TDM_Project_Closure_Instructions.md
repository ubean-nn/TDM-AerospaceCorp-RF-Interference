# TDM Project Closure Document — Instructions & Examples

> This document serves as the official record for the project, summarizing the work completed, key outcomes, challenges encountered, and lessons learned. It also confirms that project materials have been organized and shared with mentors to support future continuation of the work if needed.
>
> Teams should use the [provided template](https://the-examples-book.com/crp/students/_attachments/TDM%20Project%20Closure.dotx) to complete this document. **A complete response often requires at least three sentences per section.**

---

## Section 1: Project Overview

This project, developed in partnership with **The Aerospace Corporation**, explores the application of machine learning to automate the detection and classification of radio frequency (RF) interference signals. As wireless communication technologies expand, RF interference poses increasingly disruptive challenges, particularly for systems reliant on accurate satellite-based navigation like GPS and GNSS constellations. Monitoring RF environments manually is difficult due to the volume and complexity of the data, which motivated our approach to use deep learning pipelines.

| Field                                              | Description                                                                                                    |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Project Name**                                   | RF Interference                                                                                                |
| **Corporate Partner**                              | The Aerospace Corporation                                                                                      |
| **Student Team (Development Team)**                | Eubene In, Ashmit Tendolkar, Vineet Rao, Rex Wu, Ronak Mohanty, Adarsh Rangayyan, Tushar Bhat, Sreya Etherajan |
| **Teaching Assistant (Scrum Master)**              | William Yu                                                                                                     |
| **Corporate Partner Mentor(s) (Product Owner(s))** | Kyle Logue, Anson Lim                                                                                          |
| **Project Semester(s)**                            | Fall 205, Spring 2026                                                                                          |

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

We developed multiple distinct ML pipelines and successfully constructed an optimized 11-model ensemble. The models learned to independently evaluate processed PSD matrices to output a predicted interference class. Our total ensemble achieved a test set accuracy of 76.91%.

---

### 2.4 Business Impact

Manually monitoring RF environments is difficult due to the sheer volume and complexity of the signal data. This project demonstrates that machine learning is a promising, scalable approach for automating this detection. Ultimately, it lays the groundwork for systems that support more automated RF monitoring in real-world environments, protecting satellite-based navigation systems (like GPS/GNSS) against safety risks from jamming.

---

## Section 3: Key Achievements

Documents the final outputs produced by the team. Focus on the **most important results**, not every task completed.

### 3.1 Final Deliverables

- **`ine` / `ebroyles` / `wurex`**: Extensive implementations of deep learning (Audio Spectrogram Transformers, Hierarchical CNNs) alongside classical ML benchmarks (scikit-learn + SMOTE).
- **`tbhat` / `arangayy`**: High-performance cluster configurations, GPU evaluation routines, base PyTorch training workloads, and memory sweepers. 
- **`vrao` / `mohantr`**: Finalized ensemble voting framework pipelines, data augmentation schemes addressing heavy class imbalance, and predictions logging.
- **`atendolk` / `clubbers`**: Visualization of datasets, `radiomana` modeling pipelines, and target extraction implementations.

---

### 3.2 Code Repository

The code development has been successfully aggregated, with experimental and primary pipelines maintained under distinct modular sandboxes within the `workspaces/` directory.

`https://github.com/ubean-nn/TDM-AerospaceCorp-RF-Interference`

---

### 3.3 Data & Reports Location

The original Highway2 GNSS Jamming Dataset can be found at: `https://gitlab.cc-asp.fraunhofer.de/darcy_gnss/fiot_highway2`

---

### 3.4 Documentation

Inline documentation has been heavily localized into the individual components. Every member sub-folder within `workspaces/` (e.g., `workspaces/ine/README.md`, `workspaces/wurex/README.md`, etc.) contains a concise `README.md` explaining their respective workflows (baseline testing, AST modeling, data pipelines). Furthermore, well-annotated `.ipynb` exploratory notebooks provide interactive technical walkthroughs.

---

### 3.5 Achievements

- Improvement in model accuracy or analytical performance
- Development of a working prototype or proof of concept
- Integration of multiple data sources
- Automation of a previously manual process
- Insights that support better decision-making

We successfully developed an ensemble approach combining multiple ML architectures. As shown below, our majority voting ensemble outperformed several of the individual architectures, culminating in an overall test set accuracy of 76.91%.

| sample_id    | Result | Eubene-AST | Eubene-CNN | Rex    | Rex meeting | Rex 3  | Ronak  | Majority   |
| :----------- | :----- | :--------- | :--------- | :----- | :---------- | :----- | :----- | :--------- |
| **Accuracy** |        | 0.7791     | 0.7949     | 0.6229 | 0.2512      | 0.2512 | 0.7145 | **0.7691** |
| 12133        | 8      | 2          | 2          | 2      | 8           | 8      | 2      | 2          |
| 12821        | 8      | 2          | 2          | 2      | 8           | 8      | 8      | 2          |
| 8239         | 7      | 2          | 1          | 0      | 7           | 7      | 2      | 2          |
| 16011        | 5      | 5          | 5          | 4      | 4           | 4      | 5      | 5          |
| 14592        | 8      | 2          | 2          | 2      | 8           | 8      | 2      | 2          |
| 2010         | 7      | 7          | 7          | 0      | 7           | 7      | 7      | 7          |
| 764          | 5      | 5          | 5          | 5      | 5           | 5      | 5      | 5          |
| 8235         | 7      | 2          | 1          | 0      | 7           | 7      | 2      | 2          |
| 2803         | 7      | 7          | 7          | 0      | 7           | 7      | 7      | 7          |

---

## Section 4: Challenges and Lessons Learned (Project Reflection)

Reflects on the project experience and documents insights for future teams. Focus on key experiences — do **not** repeat deliverables already listed above.

### 4.1 Challenges Encountered

The primary challenges were the sheer volume and complexity of the raw RF spectrum data and the severe imbalance in the Highway2 GNSS Jamming Dataset.

- Limited or inconsistent data availability
- Unexpected technical difficulties or model performance issues
- Reliable Access to GPU's

---

### 4.2 Mitigation Strategies or Resolutions

To address the data imbalance, we utilized SMOTE to generate fake samples for minority classes and implemented Weighted/Focal Loss during training. To address variations in model reliability, rather than relying on a single architecture, we employed an ensemble approach with majority voting, which made weaker models significantly stronger when aggregated. Computation speed was managed by training models using GPU resources which we found to be fastest at night when the demand were low and queue was short.

---

### 4.3 Lessons Learned

- **Model Transitioning and Environment Limitations:** Translating prototyped models built interactively in Jupyter notebooks into completely headless GPU compute scripts (using `sbatch`/`rungpu.sh`) presented significant logistical challenges affecting testing times.
- **Hierarchical Structuring:** Dividing classification objectives into simpler binary and sub-group models (such as handling classes 0-3 first) proved far more effective empirically than attempting "one-shot" multi-class prediction against the 9 categories.
- **Algorithmic Limits of Classical Models:** Early attempts to benchmark using robust classical ML methods (`sklearn` configurations with SMOTE) were instrumental in demonstrating the ultimate necessity of relying on raw spatial-temporal feature extraction (CNNs/ASTs) to manage extreme radio frequency noise.

---

## Section 5: Next Steps & Project Closure

Summarizes how the project is concluded and how results may be continued in the future. Confirms that materials are organized for mentor handover.

### 5.1 Handover Summary

- Final presentation or demonstration to mentors
- Walkthrough of the repository or documentation
- Confirmation that mentor(s) can access all project materials

We created a research poster, a 3-Minute Thesis (3MT) video, and will present our findings at the upcoming symposium.

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

- **Proper Ensembling Integration:** Finalizing and optimizing a permanent single deployment pipeline that automatically aggregates voting decisions across all distinct models seamlessly.
- **Continuing Class Imbalance Research:** Conducting rigorous subsequent analyses on synthetic data creation techniques (GANs, modified SMOTE implementations) due to persistent representation issues in the Highway2 baseline.
- **Model Pruning and Deployment Profiling:** Aggressively applying pruning techniques to decrease model byte-size and directly increase inference processing speeds, enabling real-time detection thresholds suitable for live hardware edge environments.

---

### 5.4 Sign-Off (Formal Acceptance)

This section confirms that project results and deliverables have been reviewed by the corporate partner or mentor, representing formal acknowledgement that the project work has been completed and shared.

| Role                       | Name | Date | Signature |
| -------------------------- | ---- | ---- | --------- |
| Corporate Partner Mentor   |      |      |           |
| Teaching Assistant         |      |      |           |
| Team Lead / Representative |      |      |           |

---

### 5.5 Additional Notes _(Optional)_

Any other important information that does not fall under the sections above.

---
