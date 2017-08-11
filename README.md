## DEGREE (Deep Graph Embedding for Structure Learning)

These scripts are implementations of DEGREE model in our manuscript entitled "Deep Graph Embedding for Structure Learning in E-commerce". Here is the architecture of DEGREE. With DEGREE, our goal is to learn valuable network representations, as well as preserve both inter-group structure and intra-group structure in e-commerce network. 

## DEGREE architecture <a name=DEGREE_architecture> </a>

<img src="https://github.com/HKongTeam/DEGREE/blob/master/multi-DNN.jpg" width="600px" height = "300px">

## Dependencies

The pipeline requires:

* python 2.7 
* [Tensorflow package](https://www.tensorflow.org) (e.g. 1.2.0) 
* Numpy (e.g. 1.13.0)

## Tutorial
You can type python DEGREE.py to see options as follows:
Usage: DEGREE.py [options]
```
Options:
  -h, --help            show this help message and exit
  --train_file=FILE, --train_set_list=FILE
                        Required. Absolute path. Training data set (for now*.txt)
  --test_file=FILE, --test_set_list=FILE
                        Required. Absolute path. Test data set (for now *.txt)
  --nB=NB, --num_of_buyer=NB
                        Required. Num of buyer.
  --nS=NS, --num_of_seller=NS
                        Required. Num of seller.
  --outdir=STRING, --outdir=STRING
                        Required. Output folder
  --model_file=MODEL_FILE, --model_file=MODEL_FILE
                        True or False. load model from a saved file.
  --embedding_size=EMBEDDING_SIZE, --embedding_size=EMBEDDING_SIZE
                        Embedding size. Default is 16.
  --batch_size=BATCH_SIZE, --batch_size=BATCH_SIZE
                        Batch size. Default is 10000.
  --train_nbatch=TRAIN_NBATCH, --train_nbatch=TRAIN_NBATCH
                        Train n batches. Default is 5000.
  --test_nbatch=TEST_NBATCH, --test_nbatch=TEST_NBATCH
                        Test n batches. Default is 231.
  --omegaB=OMEGAB       Regularizer of Gb. Default is 1.
  --omegaS=OMEGAS       Regularizer of Gs. Default is 1.
  --lambdaB=LAMBDAB     Regularizer of Embedding. Default is 0.0003.
  --lambdaS=LAMBDAS     Regularizer of Embedding. Default is 0.01.
  --alpha=ALPHA         Regularizer of Weights. Default is 0.1.
  --lr=LR               Learning rate. Default is 0.002.
  --other_ext=OTHER_EXT, --other_ext=OTHER_EXT
                        other_extensions
```
Run following commands to learn graph embedding:
```
python DEGREE.py --train_file=../data/small_sample_train.txt --test_file=../data/small_sample_test.txt --outdir=../result --nB=100000 --nS=66020 --train_nbatch=10000 --alpha=0.1 --lr=0.01 --other_ext="_sigmoid_SGD" --model_file=False
```
## Project home page

For information on the source tree, examples, issues, and pull requests, see

    https://github.com/HKongTeam/DEGREE
