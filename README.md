# DN-PINNs (Dynamically Normalized PINNs)

<!-- 
This is a TensorFlow implementation of Dynamically Normalized Physics-Informed Neural Networks, described in our paper: 

Shota Deguchi, Mitsuteru Asai: [Dynamic & norm-based weights to normalize imbalance in back-propagated gradients of physics-informed neural networks](link), ..., 2023. 

comment out 
multiple
lines
-->

This is a TensorFlow implementation of Dynamically Normalized Physics-Informed Neural Networks, described in our [paper](): 

Shota Deguchi, Mitsuteru Asai: Dynamic & norm-based weights to normalize imbalance in back-propagated gradients of physics-informed neural networks, 2023. 


## Requirements
```
pip install -r requirements.txt
```

## Usage
Please make the following directories

* `saved_weights`
* `best_weights`
* `results`

in each working directory, as required by `train.py`, `infer.py`, `utils.py`. 

To train a model, run
```
python train.py [-e EPOCHS] [-b BATCH_SIZE] [-p PATIENCE]
```
(note: `BATCH_SIZE == -1` executes full-batch training, and mini-batching is only implemented in `03_AllenCahn`)

After training, load parameters saved in `saved_weights` or `best_weights` and evaluate the model. To do so, run
```
python infer.py
```

## Device (CPU / GPU)
In general, <code>train.py</code> assumes the use of a GPU, while <code>infer.py</code> uses CPU. To train a model with CPU, simply modify the flag parameter in <code>config_gpu(flag)</code> from <code>flag=0</code> to <code>flag=-1</code>. A brief description can be found in <code>config_gpu.py</code>. 

## Variants
For comparison, we have implemented several variants:

* `PINN` - original formulation by [Raissi+2019](https://doi.org/10.1016/j.jcp.2018.10.045)
* `MA_PINN` - max-average weighting scheme proposed by [Wang+2021](https://doi.org/10.1137/20M1318043)
* `ID_PINN` - inverse-Dirichlet weighting scheme proposed by [Maddu+2022](https://dx.doi.org/10.1088/2632-2153/ac3712)
* `DN_PINN` - dynamic & norm-based weighting scheme proposed in [our paper](link)

## Citation
Please cite our paper as: 
```
@article{DEGUCHI202x,
  title={Dynamic \& Norm-based Weights to Normalize Imbalance in Back-Propageted Gradients of Physics-Informed Neural Networks},
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
