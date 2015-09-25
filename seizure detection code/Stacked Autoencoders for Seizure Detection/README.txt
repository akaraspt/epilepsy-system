This is a seizure detection system implemented with Python 2.7.

The required packages for this system are as following:
[Python package name]==[version]
backports.ssl-match-hostname==3.4.0.2
Cython==0.22
decorator==3.4.0
ipython==3.0.0
Jinja2==2.7.3
jsonschema==2.4.0
MarkupSafe==0.23
matplotlib==1.4.3
mistune==0.5.1
networkx==1.9.1
nose==1.3.4
numpy==1.9.2
pandas==0.15.2
Pillow==2.7.0
ptyprocess==0.4
pydot==1.0.28
Pygments==2.0.2
-e git://github.com/lisa-lab/pylearn2.git@c7ebe72cab2e1abc6d7bdf931728009c52478791#egg=pylearn2-master
pyparsing==1.5.6
python-dateutil==2.4.1
pytz==2014.9
PyYAML==3.11
pyzmq==14.5.0
scikit-image==0.11.2
scikit-learn==0.15.2
scipy==0.15.1
six==1.9.0
terminado==0.5
Theano==0.7.0rc2
tornado==4.1

Some of these packages depend on CUDA 7.0 and cuDNN 6.5 from NVIDIA.
Please make sure that both of them are installed before using this sytem.

The scripts used to evaluate our model are in
 - tests/sdae_train.py (for CHBMIT dataset)
 - tests/sdae_train_epilepsiae.py (for Epilepsiae dataset)
 - tests/sdae_predict.py (for CHBMIT dataset)
 - tests/sdae_predict_epilepsiae.py (for Epilepsiae dataset)

The model are specified in yaml files, which are in yaml directory.

Note:
 - We provide this source code to illustrate the algorithms and their application.
 - The CHBMIT dataset used by this system can be downloaded from https://drive.google.com/folderview?id=0B5FFGTAGXhFEfnpKdDlRMmRSVjYxQ2hSX3ZJNTd4TE5RVXpHa1l3OFpycmFaSVpNS0JweGs&usp=sharing
   - Please also change the directory of the dataset in the script accordingly
 - Depending on the weight parameters that are randomly initialized at the beginning, the results might be varied.