# Adversarially-Learned-Anomaly-Detection
ALAD (Proceedings of IEEE ICDM 2018) official code

The code for the paper ["Adversarially Learned Anomaly Detection" (authors: Houssam Zenati*, Manon Romain*, Chuan Sheng Foo*, Bruno Lecouat, Vijay Ramaseshan Chandrasekhar)](https://arxiv.org/abs/1812.02288) is now open source! 

Please reach us via emails or via github issues for any enquiries!

Please cite our work if you find it useful for your research and work.
```
@article{Zenati2018AdversariallyLA,
  title={Adversarially Learned Anomaly Detection},
  author={Houssam Zenati and Manon Romain and Chuan Sheng Foo and Bruno Lecouat and Vijay R. Chandrasekhar},
  journal={2018 IEEE International Conference on Data Mining (ICDM)},
  year={2018},
  pages={727-736}
}
```

## Prerequisites.
To run the code, follow those steps:

Download the project code:

```
git clone https://github.com/houssamzenati/Adversarially-Learned-Anomaly-Detection.git
```
Install requirements (in the cloned repository):

```
pip3 install -r requirements.txt
```


## Doing anomaly detection.

Running the code with different options

```
python3 main.py <model> <dataset> run --nb_epochs=<number_epochs> --label=<0, 1, 2, 3, 4, 5, 6, 7, 8, 9> --m=<'cross-e','fm'> --d=<int> --rd=<int> etc. 
```
Please refer to the argument parser in main.py for more details.

When using alad, please use it with --sn, --enable_early_stop and --enable_dzz. (Different options are provided to enable the ablation study).
Important: we also provide implementations of DSEBM and DAGMM methods as open source work, the results reported in our paper for those methods, however, are derived from the DAGMM paper.
