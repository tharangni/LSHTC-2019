Recursive Regularization (RR)
-----------------------------
RR is based on the works of [1]. HierCost is based on RR and I modified the base code of HierCost by adding recursive regularization. The original code for RR is available in Java (https://github.com/gcdart/MulticlassClassifier). This was done to **reproduce** results for my baseline during my master thesis research.


HierCost
--------
HierCost toolkit is a set of programs for supervised  classification for
single-label and multi-label hierarchical classification using cost sensitive
logistic regression based classifier written in python.

For additional information on how to use the software please
visit the following webpage

https://cs.gmu.edu/~mlbio/HierCost/


Dependencies
--------------

This software is written in Python language. The following are the recommended requirements.

python 3.5 +
numpy 1.9.2 +
scipy 0.15.1 +
networkx 1.9.1 
scikit-learn 0.15.2 +
pandas 0.16.2 +


Copyright and License Information
================================================================================
HierCost is primarily written and maintained by Anveshi Charuvaka (George
Mason University) and is copyrighted by George Mason University It can be
freely used for educational and research purposes by non-profit institutions
and US government agencies only. Other organizations are allowed to use
HierCost only for evaluation purposes, and any further uses will require prior
approval.

The software may not be sold or redistributed without prior approval. One may
make copies of the software for their use provided that the copies, are not
sold or distributed, are used under the same terms and conditions.

As unestablished research software, this code is provided on an ''as is''
basis without warranty of any kind, either expressed or implied. The
downloading, or executing any part of this software constitutes an implicit
agreement to these terms. These terms and conditions are subject to change at
any time without prior notice.

References
----------
[1] Siddharth Gopal and Yiming Yang. 2013. Recursive regularization for large-scale classification with hierarchical and graphical dependencies. In Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining (KDD '13), Rayid Ghani, Ted E. Senator, Paul Bradley, Rajesh Parekh, and Jingrui He (Eds.). ACM, New York, NY, USA, 257-265. DOI: https://doi.org/10.1145/2487575.2487644