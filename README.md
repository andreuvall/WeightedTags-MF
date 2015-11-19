Matrix factorization models enhanced with social tags
=====================================================

Implementation of matrix factorization models enhanced with social tags for the task of music recommendation, as presented in the following papers:

* Andreu Vall, Marcin Skowron, Peter Knees, and Markus Schedl. “[Improving Music Recommendations with a Weighted Factorization of the Tagging Activity.](http://www.cp.jku.at/research/papers/Vall_etal_ISMIR_2015.pdf)” In Proc. ISMIR. Málaga, Spain, 2015.

* Andreu Vall. “[Listener-Inspired Automated Music Playlist Generation.](http://www.cp.jku.at/research/papers/Vall_RecSys_2015.pdf)” In Proc. RecSys. Vienna, Austria, 2015.


### Models implemented

This project implements the following models:

* MF: Matrix factorization for implicit feedback as in Yifan Hu, Yehuda Koren and Chris Volinsky, “[Collaborative Filtering for Implicit Feedback Datasets.](http://yifanhu.net/PUB/cf.pdf)” in Proc. ICDM 2008.

* TMF: Matrix factorization enhanced with social tags. Greatly based on Yi Fang and Luo Si, “[Matrix Co-Factorization for Recommendation with Rich Side Information and Implicit Feedback.](http://www.cse.scu.edu/~yfang/hetrec11-fang.pdf)” in Proc. HETREC 2011, but modified in our paper to specifically deal with tags.

* WTMF: Matrix factorization enhanced with weighted social tags as introduced in our papers (see above).


### Usage

Each model is implemented in a separated python file. You can simply run a model, say WTMF, by doing:

```bash
python WTMF.py
```

This will apply the model on the dataset specified within the `WTMF.py` file. Note that a dummy dataset is already included in the project. You can also specifiy the dataset manually, but please note the data folder structure:

```bash
python WTMF.py <collection-name> <dataset-name>
```

### Details

Specific parameters for the model need to be provided within each model file. A detailed description of the parameters can be found in the papers.

Each program outputs a json file with the expected percentile rank achieved by the model using the given dataset under the given parameters. A detailed description of the evaluation methodology can be found in the papers.

### License

The contents of this repository are licensed. See the LICENSE file for further details.
