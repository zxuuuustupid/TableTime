# TableTime: An Agentic Framework for Time Series Classification via LLM-driven Code Generation

This project introduces **TableTime**, a novel, training-free framework that reframes Multivariate Time Series (MTS) classification. We reformulate the task as a two-stage reasoning process executed by a Large Language Model (LLM) acting as an autonomous agent. Instead of inefficiently feeding raw numerical data into the LLM's context window, our approach leverages the model's advanced code generation and logical inference capabilities to achieve high-performance, interpretable, and token-efficient classification.

---

## Motivation

Directly applying Large Language Models to time series classification presents significant challenges: the high dimensionality of numerical data leads to prohibitive token consumption, and the LLM's reasoning abilities are underutilized when processing raw signals. Traditional methods, in turn, struggle to model complex inter-channel dependencies and require extensive domain expertise.

TableTime addresses these limitations by treating the LLM not as a data processor, but as a **Data Scientist Agent**. This agent autonomously analyzes the data by writing and executing code, mirroring the workflow of a human expert.

---

## Methodology

Our framework operates through a two-stage, **Coder-Analyst** workflow. This division of labor allows each LLM instance to focus on a task it excels at: one for code-based analysis, and the other for evidence-based reasoning.

### Phase 1: Autonomous Feature Engineering (The "Coder" Agent)

The first phase is designed to transform low-level, high-volume signal data into high-level, low-volume descriptive evidence.

1.  **Contextual Understanding**: We provide the Coder agent with crucial metadata, including:
    *   **Data Pointers**: File paths to the test data, training data, and a pre-computed nearest neighbor map.
    *   **Structural Schema**: The shape and format of the data arrays (`(N, Channels, Time)`).
    *   **Domain Knowledge**: A rich description of the physical system, including the meaning of each sensor channel (e.g., "CH11-13: Gearbox Input Shaft Vibration").

2.  **Autonomous Code Generation**: Armed with this context, the Coder agent autonomously determines the most effective analysis strategy. It writes a self-contained Python script to:
    *   **Select Features**: Based on the domain knowledge (e.g., inferring that "Gearbox" data requires Kurtosis for shock detection), it selects appropriate time-domain and frequency-domain features.
    *   **Perform Comparative Analysis**: The script iterates through each test sample, calculates its feature vector, retrieves its neighbors' data, calculates their feature vectors, and quantifies the mathematical similarities and differences.
    *   **Synthesize a Narrative**: The script's final output is not raw numbers, but a **"Descriptive Narrative"**. This text-based summary for each sample synthesizes numerical evidence, comparison trends, and physical meaning into an objective, human-readable report (e.g., *"Test sample exhibits a Kurtosis of 15.2... a pattern often associated with bearing defects."*).

3.  **Local Execution**: This generated script is executed locally, with the resulting narratives saved to a JSON file. This critical step ensures that the massive raw data never leaves the local environment and is not consumed as tokens.

### Phase 2: Contextual Diagnosis (The "Analyst" Agent)

The second phase uses the high-quality, condensed information from Phase 1 to make the final classification.

1.  **Evidence-Based Reasoning**: The descriptive narrative for each test sample is fed, one by one, to a second LLM instance—the Analyst.

2.  **Neighbor-Assisted Inference**: Alongside the narrative, the Analyst is also provided with the ground-truth labels of the corresponding neighbor samples. This provides strong contextual clues.

3.  **Final Classification**: The Analyst's task is to perform the final diagnosis based on the rich, pre-digested evidence. For example, if the narrative states "strong signal consistency" and the neighbors are all labeled "Health", the classification is straightforward. If the narrative highlights "significant deviation" and the neighbors are mixed, the Analyst must weigh the evidence to make a judgment.

This two-stage process transforms a complex numerical problem into a simple, evidence-based reasoning task, which is precisely what LLMs are designed to solve.

---

## Citation

If you find this work helpful for your research, please consider citing our paper:

```bibtex
@article{Duan2025TableTime,
  title={TableTime: An Agentic Framework for Time Series Classification via LLM-driven Code Generation},
  author={Zhixu Duan, Zuoyi Chen, et al.},
  journal={Pattern Recognition},
  year={2025}
}
```