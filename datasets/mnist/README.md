## PolyLUT on the MNIST dataset

To reproduce the results in our paper follow the steps below. Subsequently, compile the Verilog files using the following settings (utilize Vivado 2020.1, target the xcvu9p-flgb2104-2-i FPGA part, use the Vivado Flow_PerfOptimized_high settings, and perform synthesis in the Out-of-Context (OOC) mode).

```
python train.py --arch hdr
python neq2lut.py --arch hdr --checkpoint ./test_0/best_accuracy.pth --log-dir ./test_0/verilog/ --add-registers --seed 984237 --device 0
```


## Citation
Should you find this work valuable, we kindly request that you consider referencing our paper as below:
```
@INPROCEEDINGS{polylut,
  author={Andronic, Marta and Constantinides, George A.},
  booktitle={2023 International Conference on Field Programmable Technology (ICFPT)}, 
  title="{PolyLUT: Learning Piecewise Polynomials for Ultra-Low Latency FPGA LUT-based Inference}", 
  year={2023},
  pages={60-68},
  doi={10.1109/ICFPT59805.2023.00012}}
```