# PolyLUT
PolyLUT is the first quantized neural network training methodology that maps a neuron to a LUT while using multivariate polynomial function learning to exploit the flexibility of the FPGA soft logic. This project is a derivative work based on LogicNets (https://github.com/Xilinx/logicnets) which is licensed under the Apache License 2.0.

## Setup
**Install Vivado Design Suite**

**Create a Conda environment**

Requirements:
* python=3.8
* pytorch==1.4.0
* torchvision

## Install Brevitas
```
conda install -y packaging pyparsing
conda install -y docrep -c conda-forge
pip install --no-cache-dir git+https://github.com/Xilinx/brevitas.git@67be9b58c1c63d3923cac430ade2552d0db67ba5
```

## Install PolyLUT package
```
cd PolyLUT
pip install .
```
## Install wandb + login
```
pip install wandb
wandb.login()
```
## Summary of major modifications from LogicNets
* The novelty of our work comes through the expansion of the feature vector x at each neuron with all the monomials up to a parametric degree. When degree = 1, PolyLUT’s behavior becomes linear and it is equivalent to LogicNets. It is thus a strict generalization of LogicNets.
* Both PolyLUT and LogicNets are capable of training on the Jet Substructure Tagging and Cybersecurity (UNSW-NB15) datasets. Additionally, PolyLUT offers compatibility with the MNIST dataset.
* Introducing novel model architectures, PolyLUT's distinct structures are detailed in our accompanying paper.
* PolyLUT is tailored for optimal GPU utilization.
* To track experiments PolyLUT uses WandB insetad of TensorBoard.
* While LogicNets enforces an a priori sparsity by utilizing a weight mask that deactivates specific weights, PolyLUT takes a different approach. It doesn't employ a weight mask but rather utilizes a feature mask (FeatureMask), which reshapes the feature vector to incorporate only fanin features per output channel. We also introduce a new mask named PolyMask, facilitating expansion with all monomials.
* PolyLUT introduces a completely new forward function.
* The function "calculate_truth_tables" was adapted to align with the PolyLUT neuron structure, and it was also improved for efficiency.

## Citation
Should you find this work valuable, we kindly request that you consider referencing our paper as below:
```
@misc{andronic2023polylut,
      title={PolyLUT: Learning Piecewise Polynomials for Ultra-Low Latency FPGA LUT-based Inference}, 
      author={Marta Andronic and George A. Constantinides},
      year={2023},
      eprint={2309.02334},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```