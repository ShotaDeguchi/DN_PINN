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
@article{Deguchi_2023,
	doi = {10.1088/2399-6528/ace416},
	url = {https://dx.doi.org/10.1088/2399-6528/ace416},
	year = {2023},
	month = {jul},
	publisher = {IOP Publishing},
	volume = {7},
	number = {7},
	pages = {075005},
	author = {Shota Deguchi and Mitsuteru Asai},
	title = {Dynamic &amp; norm-based weights to normalize imbalance in back-propagated gradients of physics-informed neural networks},
	journal = {Journal of Physics Communications},
	abstract = {Physics-Informed Neural Networks (PINNs) have been a promising machine learning model for evaluating various physical problems. Despite their success in solving many types of partial differential equations (PDEs), some problems have been found to be difficult to learn, implying that the baseline PINNs is biased towards learning the governing PDEs while relatively neglecting given initial or boundary conditions. In this work, we propose Dynamically Normalized Physics-Informed Neural Networks (DN-PINNs), a method to train PINNs while evenly distributing multiple back-propagated gradient components. DN-PINNs determine the relative weights assigned to initial or boundary condition losses based on gradient norms, and the weights are updated dynamically during training. Through several numerical experiments, we demonstrate that DN-PINNs effectively avoids the imbalance in multiple gradients and improves the inference accuracy while keeping the additional computational cost within a reasonable range. Furthermore, we compare DN-PINNs with other PINNs variants and empirically show that DN-PINNs is competitive with or outperforms them. In addition, since DN-PINN uses exponential decay to update the relative weight, the weights obtained are biased toward the initial values. We study this initialization bias and show that a simple bias correction technique can alleviate this problem.}
}
```

## License
MIT License
