# Advancing Model Evaluation: Reducing Test Data Requirements for Regression with Small Datasets

<p align="left">
<b> Abstract </b>
<br>
The scarcity of large labeled datasets in many resource-constrained fields presents
significant challenges for model training and evaluation. While previous efforts
have mainly focused on reducing training data requirements, a critical and often-
overlooked challenge remains: minimizing the amount of labeled test data needed
to accurately assess model performance. Standard uniform sampling techniques,
employed to select points for labeling from a pool of unlabeled test candidates,
demand substantial test set sizes to guarantee robust generalization estimates.
Nonuniform methods, though attractive, often introduce bias, compromising
accurate performance evaluations. To address this gap, this work extends a recent
surrogate sampling framework for regression evaluation by proposing four new
surrogate models with nonuniform sampling distributions. These models encom-
pass a spectrum of complexity, from deterministic neural networks (e.g., Monte
Carlo dropout) to those leveraging probabilistic layers or weights (e.g., mean-field
variational inference). Consequently, this work facilitates a thorough investiga-
tion of the tradeoffs between surrogate complexity and test data requirements.
Experimental results on synthetic and real-world datasets show that our approach
can reduce the test data requirements to as little as 10% of the original test set
while achieving the same test error as the full test set. In comparison, uniform
sampling typically requires over 50% of the test data to achieve similar perfor-
mance. By integrating diverse surrogate models into the nonuniform sampling
framework, we significantly improve data efficiency in machine learning model
evaluation and provide insights into training run requirements and computational
costs.</p><br>

<b>Keywords:</b> Data scarcity, Model evaluation, Active testing, Surrogate models, Computational efficiency

<br>
<br>
  The primary contributions of this work are summarized below, organized by the sampling technique employed:
  
  1. Deterministic Neural Network with Bootstrap Sampling: Building upon the bootstrap technique, this approach extracts stochastic out-puts from a deterministic neural network by training the surrogate model on multiple bootstrap samples of the training dataset. While the deterministic neural network requires minimal assumptions, generating multiple bootstrap datasets and refitting the surrogate model imposes significant computational overhead. This method is particularly useful in low-dimensional settings with limited data, where resampling improves the representativeness of the test set and enhances uncertainty quantification.

2. Deterministic neural network with Monte Carlo dropout layer: This approach extends the deterministic architecture by incorporating Monte Carlo (MC) dropout during inference. Traditionally, dropout is applied during training to randomly deactivate neurons, while all neurons remain active during inference. MC dropout applies dropout layers at inference time, offering an inexpensive approximation of Bayesian neural networks by introducing stochasticity into an otherwise deterministic model. Since the model only needs to be trained once, this approach improves computational efficiency while still capturing meaningful model uncertainty with minimal assumptions.

3. Stochastic neural network with probabilistic output layer: In this approach, the deterministic neural network is appended with a probabilistic out-
put layer from the TensorFlow Probability library. By leveraging the built-in stochasticity of probabilistic layers, this method eliminates the need for repeated
surrogate model refitting, significantly reducing computational costs when constructing the sampling distribution. This approach is particularly beneficial when
dealing with large pools of unlabeled data. However, the increased stochasticity can lead to additional variance in uncertainty estimates, reducing the stability of test
loss estimates compared to deterministic sampling techniques.

4. Bayesian neural network with mean-field variational inference: This approach integrates variational inference (VI) through TensorFlow Probability’s
framework. VI is particularly effective in high-dimensional, complex data scenarios with large pools of unlabeled data. While variational inference reduces the
computational burden associated with estimating the sampling distribution, its effectiveness depends on proper calibration and sensitivity to underlying model
assumptions. This method offers a more principled and scalable alternative when compared to traditional sampling techniques.

<br>

Compatibility for TFP and Keras_Tuner have been compromised with recent updates.  Use the following environment for successful run of files: Scikit-learn v1.4; Numpy v1.26.4; Pandas v2.2.1; Matplotlib v3.9; Keras v2.15; TF v2.15; TFP v0.23; KT v1.4.7.

