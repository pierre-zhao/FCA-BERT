# FCA-BERT
This repo provides the code for reproducing the experiments in ACL-2022 paper: [Fine- and Coarse-Granularity Hybrid Self-Attention for Efficient BERT](https://arxiv.org/pdf/2203.09055.pdf). This code is adapted from the repos of  [PoWER-BERT](https://github.com/IBM/PoWER-BERT).



## Environment
tensorflow-gpu==1.15.0 <br>
keras==2.3.0 <br>
keras_bert==0.60.0  


## Dataset
Before running this Repo you should download the [GLUE](https://gluebenchmark.com/tasks) data and then use this [script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e) to unpack it to some directory `$GLUE_DIR`. Also, download the tf-version pre-trained checkpoint([BERT-base/large](https://github.com/google-research/bert), [ELECTRA-base/large](https://github.com/google-research/electra), Distil-BERT etc.) and unzip it to some directory `$BERT_DIR`. 

## Running
The detailed training and inference steps including the parameters are given in the run.sh.


## Citation

```
@inproceedings{zhao-etal-FCA,
    title = "Fine- and Coarse-Granularity Hybrid Self-Attention for Efficient BERT",
    author = "Zhao, Jing  and
      Wang, yifan  and
      Bao, Junwei  and
      Wu, Youzheng  and
      He, Xiaodong",
    booktitle = "ACL  2022",
    year = "2022",
    publisher = "Association for Computational Linguistics",
}

```

