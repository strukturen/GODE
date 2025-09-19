# Grammar-based Ordinary Differential Equation Discovery (GODE)

This repository contains the code for the paper 'Grammar-based Ordinary Differential Equation Discovery'. The preprint is available under https://arxiv.org/abs/2504.02630. 

GODE is a methodology for the end-to-end discovery of symbolic ordinary differential equations (ODEs). It combines formal grammars with dimensionality reduction and stochastic search to efficiently search high-dimensional combinatorial spaces. Grammars allow us to seed domain knowledge and structure in both the generation of large pretrained libraries as well as inference processes, effectively reducing the exploration space. This method has been validated on first- and second-order linear and nonlinear ODEs with examples from structural dynamics.

## Folder structure

- GODE: contains code to train the GODE and discover equations with the GODE.
- odeformer: code to discover equations with ODEFormer. It was slightly adapted to fit the purpose of the comparison and to generate own models. (https://github.com/sdascoli/odeformer)
- ProGED: code to discover equations with ProGED, used version is 0.8.5. (https://github.com/brencej/ProGED)
- PySR: code to discover equations with PySR, used version is 0.19.4. (https://github.com/MilesCranmer/PySR)
- Plots: contains csv-files of the discovered equations by all models and jupyter notebooks to visualize the results

## Getting started
To test GODE, please install the requirements of the file 'requirements.txt' in the GODE folder. 

The paper presents three benchmarks: Benchmark 1 on one-dimensional explicit ODEs, Benchmark 2 on linear and nonlinear ODEs, Benchmark 3 on nonlinear ODEs from structural dynamics and Benchmark 4 contains the Silverbox benchmark. 

For each benchmark, a separate model has been trained (besides Benchmark 3 and 4 use the same model). Depending on the benchmark, different grammars with different maximum lengths of the rule sequence are used, this needs to be specified udner 'grammar/common_args.py'. To for instance test example ID 5 from Benchmark 1 with three runs, after specifying the maximum length of the rule sequence 'MAX_LEN', run in the command line:
```
python GODE/EquationDiscovery_B1.py 5 3
```

### Disclaimer
Some code snippets from the files 'GODE/parser/cfg_parser.py', 'GODE/model/vae.py' are from the SDVAE implementation: https://github.com/Hanjun-Dai/sdvae, however, the GVAE algorithm follows the original GVAE implementation (https://github.com/mkusner/grammarVAE).

## Citing
If you are using the code or find it useful, please consider citing the preprint. The journal paper has been accepted and is currently in-press, once published we will update the citation below.
```
    @misc{yu_grammar-based_2025,
      title = {Grammar-based {Ordinary} {Differential} {Equation} {Discovery}},
      author={Yu, Karin L. and Chatzi, Eleni and Kissas, Georgios},
      url = {https://arxiv.org/abs/2504.02630},
	  doi = {10.48550/ARXIV.2504.02630},
	  publisher = {arXiv},
	  year = {2025},
    }
```