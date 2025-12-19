# Offline LLMs for Low-Spec Devices: A Study on Efficiency and Usability

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/Framework-Flask-green)
![Library](https://img.shields.io/badge/Library-HuggingFace_Transformers-yellow)
![Status](https://img.shields.io/badge/Status-Research_Prototype-orange)

##  Abstract
Artificial Intelligence (AI) language models have revolutionized NLP, but their reliance on cloud-based infrastructures poses challenges regarding privacy, accessibility, and resource dependency. This project evaluates the feasibility of running compact AI models on personal computing devices without internet access. 

We developed a full-stack Flask-based benchmarking system to measure the performance of five compact models across various NLP tasks, employing optimization techniques like quantization and pruning to ensure usability on consumer-grade hardware.

##  Key Features
* **Offline Capability:** Run LLMs locally without internet connectivity, ensuring data privacy.
* **Multi-Model Support:** Benchmarks 5 distinct compact architectures.
* **Adaptive Inference Framework:** Dynamically adjusts model parameters (e.g., batch size) based on available system resources (RAM/CPU).
* **Comprehensive Benchmarking:** Measures inference speed (ms/token), memory usage, and output quality (BLEU/ROUGE).
* **Interactive Dashboard:** A Flask + Jinja web interface to visualize performance metrics.

##  Supported Models
This study focuses on models designed for low-resource environments:
1.  **DistilBERT** (Knowledge Distillation)
2.  **MobileBERT** (Mobile-optimized)
3.  **TinyLLaMA** (Compact Generative)
4.  **Phi-2** (Microsoft)
5.  **Gemma-2B** (Google)

## 🛠️ Tech Stack
* **Backend:** Python, Flask
* **Frontend:** HTML/CSS, Jinja2 Templates
* **ML Frameworks:** PyTorch, TensorFlow, ONNX
* **Libraries:** Hugging Face Transformers, NumPy
* **Optimization:** Quantization (FP32 -> INT8), Pruning

##  System Architecture
The application follows a modular architecture optimized for low-spec devices:

1.  **Memory Optimization Layer:** Handles Quantization and Weight Sharing.
2.  **Model Execution Environment:** Loads and manages the specific model checkpoints.
3.  **Inference Engine:** Manages token processing and sequence generation.
4.  **Task Processor:** Routes requests to specific tasks (QA, Summarization, etc.).
5.  **User Interface:** Flask-based dashboard for user interaction.

## Evaluation Metrics
We evaluate models on the following criteria:
* **Inference Speed:** Milliseconds per token.
* **Memory Consumption:** Peak RAM usage during inference.
* **Quality:** Perplexity, BLEU, and ROUGE scores.
* **Datasets Used:** SQuAD (QA), CNN/DailyMail (Summarization), WikiText (MLM).

##  Contributors
* **Koppula Sricharan Sai Krishna Phani** - *Malla Reddy University*
* **J Srija Rao** - *Malla Reddy University*
* **G Surya Kiran Reddy** - *Malla Reddy University*
* * **Pujari Tharun Kumar** - *Malla Reddy University*
* **K Venakata Naga Sreedhar** - *Malla Reddy University*



---
*This project aims to democratize AI by making intelligent language models accessible to users with limited connectivity and computing power.*
