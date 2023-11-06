## PolyLUT on the network intrusion dataset (UNSW-NB15)

To reproduce the results in our paper follow the steps below. Subsequently, compile the Verilog files using the following settings (utilize Vivado 2020.1, target the xcvu9p-flgb2104-2-i FPGA part, use the Vivado Flow_PerfOptimized_high settings, and perform synthesis in the Out-of-Context (OOC) mode).
```
python train.py --arch nid-lite
python neq2lut.py --arch nid-lite --checkpoint ./test_0/best_accuracy.pth --log-dir ./test_0/verilog/ --add-registers --seed 1697 --device 0
```


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