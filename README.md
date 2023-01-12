# DN-PINNs (Dynamically Normalized PINNs)

This is a TensorFlow implementation of Dynamically Normalized Physics-Informed Neural Networks, described in our [paper](link). 

## Requirements
```
pip install -r requirements.txt
```
or
* tensorflow (2.5.0 or later)
* numpy
* scipy

## Device (CPU / GPU)
In general, <code>train.py</code> assumes the use of a GPU, while <code>infer.py</code> uses CPU. To train a model with CPU, simply modify the flag parameter in <code>config_gpu(flag)</code> from <code>flag=0</code> to <code>flag=-1</code>. A brief description can be found in <code>config_gpu.py</code>. 

## Citation
Please cite our paper as: 
```
@article{DEGUCHI202x,
  title={Dynamic & Norm-based Weights to Normalize Imbalance in Back-Propageted Gradients of Physics-Informed Neural Networks},
  author={Shota DEGUCHI and Mitsuteru ASAI},
  journal={JOURNAL XXX},
  volume={XXX},
  number={XXX},
  pages={XXX-XXX},
  year={202x},
  doi={XXX}
}
```

## License
MIT License
