# FashionMNIST AdaBatch Replication

Implementation of the adaptive batch size algorithm from:

Devarakonda, A., Naumov, M., & Garland, M. (2017). AdaBatch: Adaptive Batch Sizes for Training Deep Neural Networks. [Source Paper](https://arxiv.org/pdf/1712.02029)

Applied to the FashionMNIST dataset for EE 516: Mathematical Foundations of Machine Learning.

### AdaBatch Intro
The AdaBatch algorithm aims to alleviate the trade-offs between small and large batch sizes by adaptively increase the batch size during the training process. More specifically, "we begin with a small batch size r, chosen to encourage rapid convergence in early epochs, and then progressively increase the batch size between selected epochs as training proceeds. For the experiments reported in this paper, we double the batch size at specific intervals and simultaneously adapt the learning rate α so that the ratio α/r remains constant". 

**Note**: In my implementation I've kept all of the original values from the paper the same. 


### Run the python script

```bash
# If using venv
pip install -r requirements.txt
# run the program
python src/adabatch_fashion_mnist.py
```
