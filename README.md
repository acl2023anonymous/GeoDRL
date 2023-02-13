# Introduction

Code and data for Anonymous ACL 2023 Submission "GeoDRL: A Self-Learning Framework for Geometry Problem Solving using Reinforcement Learning in Deductive Reasoning".

In this paper, we propose **GeoDRL**, a self-learning geometry problem solving framework by integrating Deep Reinforcement Learning into deductive geometry reasoning, which enables unsupervised learning of problem-solving strategies. We structure geometry information as Geometry Logic Graph (GLG) to glue discrete geometry literals together. The combination of neural network and symbolic system allows efficient solution while maintaining mathematical correctness.

![architecture](architecture.pdf)

## Prepare the Dataset

unzip data files by running:

```shell
sh unzip_data.sh
```

## Requirements

Install python dependencies by running:

```
pip install -r requirement.txt
```

## GeoDRL

GeoDRL is consisted of two parts: parser and reasoner.

### Parser

The parser parses the question diagram and text into logical forms. The diagram parser `PGDPNet` (https://github.com/mingliangzhang2018/PGDP) extracts geometry elements and relationships from diagram. The text parser is a ruled-based semantic parser which convert question text into geometry literals.

After that, we structure the logical forms into Graph Logical Graph (GLG) by `reasoner/symbolic_system/converter.py` as the problem state.

## Reasoner

The reasoner contains a DRL agent to select geometry theorem for each problem state and a symbolic geometric system to perform theorems. We use a Graph Transformer as the Q-Net to learn representation of GLG. The graph transformer model generates probability scores as the Q-value for each theorem with an input state. The model is self-trained on problems from train set using epsilon-exploration algorithm.

Train DRL agent by running:

```python
cd reasoner/agent
python train.py --batch_size BATCH_SIZE --gamma GAMMA --beam_size BEAM_SIZE --lr LR
```

Eval the trained model and produce a result list by running:
(use ground truth parsing results)
```python 
python eval.py --use_annotated --model_path MODEL_PATH --output_path OUTPUT_PATH --beam_size BEAM_SIZE
```
(use generated parsing results)
```python 
python eval.py --model_path MODEL_PATH --output_path OUTPUT_PATH --beam_size BEAM_SIZE
```

Use `sub_acc.py` to calculate the accuracy of each problem type:
```python
python sub_acc.py --result_file RESULT_FILE 
```




