# MCKT: A Mamba Model for Content-Aware Knowledge Tracing

The implementation of the paper *MCKT: A Mamba Model for Content-Aware Knowledge Tracing.*

## Abstract
Knowledge tracing (KT) is a critical task in educational data mining, essential for accurately modeling and predicting student learning progress. However, existing KT models often overlook the rich textual information embedded in exercises, which is crucial for understanding the context and content of student interactions. To address this gap, we propose MCKT, a novel model that introduces a Content Encoder specifically designed to capture and integrate textual information from educational exercises. Additionally, given the extended sequence lengths typical in real-world KT scenarios, traditional sequence modeling approaches like RNNs and Transformers face limitations, including issues with long-term dependency modeling and computational inefficiency. To overcome these challenges, we incorporate the Mamba module into our model, leveraging its state space modeling capabilities to more effectively handle long sequences with greater efficiency. Extensive experiments on both traditional and content-based datasets demonstrate that MCKT significantly outperforms state-of-the-art models, providing more accurate and context-aware predictions of student performance.

## Overall Architecture
*<img width="1480" alt="model structure_1" src="https://github.com/user-attachments/assets/c437c0e4-1342-4a50-9ef5-6e05afd9c48f">*

## Models

- `/KnowledgeTracing/model/Model.py`: Multiple models; selection in `main.py`.
- `/KnowledgeTracing/data/`: Reading and processing datasets.
- `/KnowledgeTracing/evaluation/eval.py`: Calculate losses and performance.

## Setup

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

### Run
```bash
run main.py
```

## Future Release
More details will be released shortly.
