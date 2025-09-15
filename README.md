# Continual Learning with Surprise Replay & Dual Learner

This repository implements methods for reducing catastrophic forgetting in large language models (LLMs) via *surprise-based replay* and a *dual learner architecture*.  

---

## What’s in This Code

- **Surprise Replay**: a replay buffer that retains the *most surprising* sequences rather than using uniform sampling.  
- **Dual Learner Architecture**: fast & slow LoRA heads, where the slow learner consolidates knowledge from the fast learner via exponential moving average (EMA).
- Variants / ablations:  
  1. Surprise computed on full sequences vs. just labels  
  2. Surprise & buffer update before vs. after training on each dataset  
  3. Reservoir buffer
- Benchmarks: experiments on the *Standard CL Benchmark* and *Large Number of Tasks* settings.  

---

## Installation & Requirements

```bash
# Clone the repository
git clone https://github.com/HHazard2002/continual-learning.git
cd continual-learning

# Python environment (example)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# continual-learning
```
---

## Repository structure
```
├── configs/                # Configurations for LoRA and Trainer
│   ├── lora_config.py   
│   ├── training_config.py
├── methods/                # Implementations for various continual learning approaches
│   ├── buffer.py   
│   ├── dual_learner.py
│   ├── ewc.py   
│   ├── replay.py
│   ├── surprise_replay.py
├── scripts/                # Scripts that can be used to reproduce experiments
│   ├── mlt.sh
│   ├── train.sh
├── utils/                  # Additional functions which are used to load and pre-process the data, prepare the LLMs, etc...
│   ├── datasets.py         # Loads and prepares the datasets  
│   ├── evaluation.py       # Used to evaluate the model's performance
│   ├── inference.py        # Used to generate text during evaluation
│   ├── metrics.py          # Computes metrics for surprise and performance
│   ├── model.py            # Prepare the LoRA heads
│   ├── script.py
├── main.py                 # The main file that runs the training pipelines
├── README.md               # This file
```
