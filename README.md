
# SSDL

This repository contains scripts for decoding emotion states from intracranial neural activity.

This repository hosts the code for the following manuscript: *Cross-task, explainable, and real-time decoding of human emotion states by integrating grey and white matter intracranial neural activity*.

# Code structure

We designed a self-supervised learning model structure that mainly consists of the following components: (1) a supervised F-statistic-driven method to identify the top discriminative neural features; (2) an unsupervised long short-term memory (LSTM) autoencoder to capture nonlinear neural dynamics through low-dimensional representations that enable one-step-ahead prediction of future neural features; (3) supervised multi-layer perceptron (MLP) modules to use the low-dimensional representations to predict (via regression) or categorize (via classification) valence and arousal ratings. 


<p align="center"><img width="782" height="257" alt="fig2" src="https://github.com/user-attachments/assets/06eaf916-f466-4e87-b9b5-9796d1fbcca2" /></p>



**(1) Supervised feature selection.** Our model first used supervised, F-statistic-driven feature selection to identify the top discriminative neural features. We computed the F-statistics through sequential linear regression between time-averaged neural features and valence/arousal ratings across training trials. Then we converted F-values to P-values using the Satterthwaite approximation with FDR correction. Finally, we ranked and selected the top d_F features by ascending P-value.

**(2) Unsupervised LSTM autoencoder.** We next used an LSTM autoencoder to capture nonlinear neural dynamics through low-dimensional representations. The LSTM encoder compressed the high-dimensional neural feature inputs into low-dimensional latent representations while preserving critical temporal dynamics. The LSTM decoder was designed to predict one-step-ahead future neural features rather than reconstruct the input, enforcing the encoder to learn temporally informative latent representations.

**(3) Supervised emotion state prediction.** After training the LSTM autoencoder, we only used the encoder part to compute the hidden state h_t at each time step t. We then averaged the hidden representations across all time steps within the trial as s_n. We then related the hidden representation s_n  to the valence/arousal rating in that trial, i.e., y_n, via a two-layer multi-layer perceptron (MLP) regressor for fine-grained prediction of continuous valence/arousal ratings. To classify coarse-grained discretized valence and arousal states, we first transformed the continuous valence or arousal ratings into discrete categories. Similar to the emotion state prediction decoder, we used a two-layer MLP classifier. 

Scripts for model construction, fitting, prediction, evaluation, and visualization are included in the [./model](./model)


# Licence
Copyright (c) 2025 Zhejiang University
