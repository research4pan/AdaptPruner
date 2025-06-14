# Adapt-Pruner

Adapt-Pruner is a structured pruning method, exploits the skewed importance distribution across LLM layers and significantly outperforms conventional pruning methods in commonsense benchmarks.

## Features

- Efficient layer-wise adaptive pruning
- A novel acceleration paradigm called Adapt-Accel, which is the first method that interleaves the pruning with training in a highly frequent manner
- Easy to use Python APIs

## Installation

```bash
git clone https://github.com/research4pan/AdaptivePruning-LLM -b beta
```

Install dependent packages

```bash
pip install -r requirements.txt
```

**❗Then please replace the corresponding files of Transformers package using the code in custom_transformer_package**


## Reproduction Results
We provide scripts to prune:

1. MobileLLM-350M → 125M
2. MobileLLM-600M → 350M
3. MobileLLM-1B → 600M
4. Qwen-2.5-0.5B → 350M
5. Deepseek-R1-Distill-Qwen-1.5B → 1B   


To run MobileLLM-350M → 125M experiment, the first step is to run processing dataset as:
```bash
bash process_dataset.sh bash process_datasets.sh MobileLLM-300MB PROCESS_DATA_DIR
```
And then prune the model as:
```bash
bash iterative_Prune_Train_Qwen0.5B.sh MODEL_DIR DATA_DIR
```
change MODEL_DIR to where you want to save the model, and DATA_DIR to the directory of processed data. And the final output model will be saved in output_model folder. The model in output_model folder can be used for the evaluation.
Similar scripts for pruning other models are provided in the [experiments](experiments/) folder.

To do few-shots evaluation using TruthfulQA, AGIEval, and MMLU, run the evaluation script as:
```bash
bash eval_fewshots.sh MODEL_DIR EVAL_LOG_DIR 
```

To do zero-shot evaluation and Wikitext2 Perplexity, run the evaluation script as:
```bash
bash eval_zeroshot.sh MODEL_DIR EVAL_LOG_DIR 
```

Notice, MODEL_DIR should be the output_model directory in previously saved model files.

## Evaluation results
Following the above evaluation steps, the MMLU results are
|Model Name|MMLU|
|----|----|
|MobileLLM-350M → 125M|25.35|
|MobileLLM-600M → 350M|32.25|
|MobileLLM-1B → 600M|37.34|

## Examples

For detailed usage examples and tutorials, please refer to the [experiments](experiments/) in the repository.

## License

Apache-2.0 license

## Acknowledgments

This project is built on top of [LLM-Pruner](https://github.com/horseee/LLM-Pruner), an efficient pruning library for LLM. We thank the LLM-Pruner team for providing this foundation.

## Citation
```
@article{boyao-adappruner,
  title={Adapt-Pruner: Adaptive Structural Pruning for Efficient Small Language Model Training}, 
  author={Boyao Wang and Rui Pan and Shizhe Diao and Xingyuan Pan and Jipeng Zhang and Renjie Pi and Tong Zhang},
  year={2025},
  journal={arXiv preprint arXiv:2502.03460}
}
```
