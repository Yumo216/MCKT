# MCKT: A Parallel Mamba-based Model for Content-Aware Knowledge Tracing

The implementation of the paper *MCKT: A Parallel Mamba-based Model for Content-Aware Knowledge Tracing.*

## Abstract
Knowledge tracing (KT) aims to model students’ evolving mastery of concepts, forming a cornerstone of intelligent tutoring systems and adaptive learning technologies. Despite recent advances, many existing KT models primarily rely on explicit exercise attributes, overlooking the rich contextual cues hidden in exercise content, including implicit textual information and cognitive difficulty, which limits both predictive performance and interpretability. Moreover, effectively capturing long-term dependencies in student learning trajectories remains a challenge, especially in the presence of long or irregular interaction sequences. To address these challenges, we propose MCKT, a novel parallel Mamba-based knowledge tracing model that jointly models the semantic structure of exercise texts and the dynamic complexity of difficulty levels. Specifically, MCKT incorporates a pre-trained language model to extract fine-grained semantic features from exercises. To model both student responses and perceived difficulty sequences, MCKT employs two parallel sequence encoders, each built upon the Mamba architecture, to efficiently model the temporal dynamics of student responses and perceived difficulty sequences. Furthermore, to improve generalization for students with sparse learning records, which are common in practical long-tail scenarios, we introduce an adaptive enhancement module that better captures short sequence patterns. Extensive experiments on benchmark datasets show that MCKT achieves consistent gains in predictive performance and robustness, supported by Mamba’s capability to efficiently capture long-range dependencies in student learning trajectories. This content-aware design enhances interpretability by mapping learning behaviors to meaningful textual and cognitive aspects of exercises, enabling more transparent and adaptive educational support.

## Overall Architecture
<img width="1710" alt="model structure_1" src="https://github.com/user-attachments/assets/722011b4-dd4a-4bcc-8be5-619ae7d901db" />


## Datasets
Evaluate our models on six benchmark datasets for knowledge tracing, as mentioned in the paper. All datasets are available in the Google Drive, which can be accessed via the following link:

 [Google Drive](https://drive.google.com/drive/folders/15rkZiXMr6IBoJr_15v5Rh82gKS6P1r45?usp=drive_link)



### Requirements

- Python 3.7+
- PyTorch 1.12+
- CUDA 11.6+

### Install Dependencies

To install `causal Conv1d` and the core `Mamba` package, run:

```bash
pip install causal-conv1d>=1.2.0
pip install mamba-ssm
```
If you encounter any issues during the installation of **Mamba**, please refer to the following GitHub discussion for possible solutions:

👉 [state-spaces/mamba#169](https://github.com/state-spaces/mamba/issues/169)

### Run
```bash
run main.py
```

