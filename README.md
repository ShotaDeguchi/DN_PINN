# DN-PINNs (Dynamically Normalized PINNs)

This is a TensorFlow implementation of Dynamically Normalized Physics-Informed Neural Networks, described in our paper: 

Shota Deguchi, Mitsuteru Asai: Dynamic & norm-based weights to normalize imbalance in back-propagated gradients of physics-informed neural networks, *Journal of Physics Communications*, 2023 (doi: [https://doi.org/10.1088/2399-6528/ace416](https://doi.org/10.1088/2399-6528/ace416))

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
(note: `BATCH_SIZE == -1` (default) executes full-batch training, and mini-batching is only implemented in `03_AllenCahn`)

After training, load the parameters stored in `saved_weights` or `best_weights` and evaluate the model. To do this, run
```
python infer.py
```

## Device (CPU / GPU)
<code>train.py</code> assumes the use of a GPU, while <code>infer.py</code> uses the CPU. To train a model on the CPU, simply change the flag parameter in <code>config_gpu(flag)</code> from <code>flag=0</code> to <code>flag=-1</code>. A short description can be found in <code>config_gpu.py</code>. 

## Variants
For comparison, we have implemented several variants:

* `PINN` - original formulation by [Raissi+2019](https://doi.org/10.1016/j.jcp.2018.10.045)
* `MA_PINN` - max-average weighting scheme proposed by [Wang+2021](https://doi.org/10.1137/20M1318043)
* `ID_PINN` - inverse-Dirichlet weighting scheme proposed by [Maddu+2022](https://dx.doi.org/10.1088/2632-2153/ac3712)
* `DN_PINN` - dynamic & norm-based weighting scheme proposed in our paper
* `DN_PINN_Bias_Corr` - dynamic & norm-based weighting scheme with bias correction proposed in our paper

## Citation
Please cite our paper as: 
```
@article{Deguchi2023dnpinn,
	author={Deguchi, Shota and Asai, Mitsuteru},
	title={Dynamic and norm-based weights to normalize imbalance in back-propagated gradients of physics-informed neural networks},
	journal={Journal of Physics Communications},
	url={http://iopscience.iop.org/article/10.1088/2399-6528/ace416},
	year={2023},
	doi={https://doi.org/10.1088/2399-6528/ace416}
}
```

## License
MIT License
