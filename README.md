# Advancing Model Evaluation: Reducing Test Data Requirements for Regression with Small Datasets

<p align="center">
<b> Abstract </b>
<br>
While deep learning continues to dominate neural network applications, many fields struggle with the impracticality of acquiring large labeled datasets due to
cost and resource constraints. Previous efforts have largely focused on reducing training data requirements, but a critical and often-overlooked challenge remains:
minimizing the amount of test data needed to accurately assess model performance. Conventional testing methods introduce bias, necessitating large test sets
to ensure reliable generalization estimates. This paper introduces a paradigm shift in model evaluation by leveraging a newly developed unbiased sampling
distribution to construct four novel sampling frameworks. These frameworks dramatically reduce test data requirements while improving test error accuracy and
reducing variance. In real-world applications, we demonstrate that our approach slashes test data needs to just 10% of the original test dataset—yet achieves the
same test error as using the full test set. Compared to standard uniform sampling, which typically demands over 50% of the test dataset, our method represents a
major advancement in efficiency. This breakthrough has the potential to redefine model validation strategies across data-scarce domains, enabling more robust AI
solutions with drastically lower data demands.</p><br>

<b>Keywords:</b> Regression modeling, Test data reduction, Small datasets, Scarce data, Unbiased Sampling, Model evaluation, Generalization error, Efficient testing, Neural
networks, Data-efficient AI, Sampling frameworks, Bias reduction, Test error estimation, Active Testing
<br>
<br>
The main contributions of this work are detailed below:

\begin{enumerate}
    \item \textbf{Extending Active Testing with Data-Driven Sampling} -- This work extends recent advancements in active testing, which introduced a corrective weighting scheme to define an unbiased estimator of test error. Building upon this foundation, we propose novel, data-driven sampling frameworks that select test sets with diverse and informative samples. These frameworks are designed using both deterministic and stochastic neural network architectures, and our four new architectures can be integrated into a unified, flexible sampling framework.   
    \item \textbf{Deterministic and Stochastic Sampling Techniques} -- The first two techniques leverage a deterministic neural network architecture to implement either Bootstrap Sampling or a Permanent Dropout Layer, both of which assess sample point uncertainty during inference. The remaining two techniques introduce stochastic neural networks, which incorporate either a Probabilistic Output Layer or a Bayesian Neural Network with Mean-Field Variational Inference, further enhancing the sampling process.  
    \item \textbf{Improved Test Loss Estimation in Scarce Data Settings} -- This approach provides a powerful solution for designing nonuniform sampling distributions, effectively reducing data requirements and significantly improving the accuracy of test loss estimates in scarce data scenarios. Our results across all four architectures demonstrate a substantial reduction in the variance of test loss estimation, achieving nearly identical true test error values with very small test sets.  
    \item \textbf{Computational Efficiencies with TensorFlow Probability} -- Finally, we highlight the computational efficiencies gained by leveraging TensorFlow Probability's built-in stochastic output capabilities \cite{tf2015}, which dramatically reduce the computational cost associated with this nonuniform sampling framework by eliminating the refitting phase of the deterministic models.  
\end{enumerate}


Compatibility for TFP and Keras_Tuner have been compromised with recent updates.  Use the following environment for successful run of files: Scikit-learn v1.4; Numpy v1.26.4; Pandas v2.2.1; Matplotlib v3.9; Keras v2.15; TF v2.15; TFP v0.23; KT v1.4.7.

