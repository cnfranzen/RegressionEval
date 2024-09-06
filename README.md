# RegressionEval
Evaluation of Regression Model using Active Testing to Reduce Data Requirements


\abstract{
    \lipsum[1-1]
    }

\keywords{Predictive modeling, Neural networks, Scarce data, Active Testing, Test Sample, Sampling, Small data}

The main contributions of this work are detailed below:
\begin{enumerate}
    \item First, this paper details a novel sampling framework that uses data-driven, sampling methods to design nonuniform sampling distributions to select relevant test points for scarce data, regression problems.
    \item Additionally, we detail four difference implementations: one previously explored in \cite{pourkamali2023regression} and used to benchmark performance for three new distribution designs.
    \item Furthermore, we demonstrate in all four implementations that a more accurate test error estimation is achieved with a reduced sample size for the test set. 
    \item Finally, we demonstrate the benefits of leveraging built-in stochastic output capabilities in Tensorflow Probability \cite{tf2015} package to reduce computational time for this sampling framework.
\end{enumerate}
