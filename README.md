# IRAS & IDRAS
Code for the papers Identifying Regulation with Adversarial Surrogate (IRAS), Identifying Dynamic Regulation with Adversarial Surrogate (IDRAS)

The IRAS code is at [IRAS.ipynb](IRAS.ipynb). It is recommended to open it in Colab to have in interactive experience.
The IDRAS code is at IDRAS_Feynman.ipynb, IDRAS_Proteins.ipynb and IDRAS_for_Bacteria.ipynb. 

To run IDRAS on different datasets, the reader should copy the code-blocks of IDRAS from one of these notebooks and call IDRAS using the command 
```
// IRAS_train_script(observations, observations_tVec, V, nativeIRAS=False)
```
with observations having shape of (N x T x F) and observations_tVec (N x T x 1). Here N is the number of observed timeseries, T is the number of samples in each timeseries and F the number of observables. The observations are in the variable observations and their corresponding times in observations_tVec. In the input V the user can place an hypothesize. The Pearson correlation between the hypothesis and the output of the algorithm is printed and their figure is plotted.

Code for the paper "Analysis of the Identifying Regulation with Adversarial Surrogates Algorithm" is at IRAS_Ribosome_RQ_paper.ipynb.
