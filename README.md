MLP (Multi-Layer Perceptron) Decision Boundary: The MLP model effectively separates the two classes in the XOR dataset, demonstrating its flexibility in handling non-linear data. However, this flexibility can also lead to overfitting, where the model learns overly intricate patterns that might not generalize well to new data. This results in a more complex decision boundary that may fit the training data closely but might struggle on unseen samples.

MLP with L1 Regularization Decision Boundary: L1 regularization encourages the model to produce sparse weights by penalizing their absolute values in the loss function. This results in a simpler decision boundary, as the regularization process suppresses less important weights, reducing the likelihood of overfitting. In the XOR dataset, this leads to a cleaner, less complex decision boundary, which may come at the expense of missing some finer details of the dataset.

MLP with L2 Regularization Decision Boundary: L2 regularization discourages large weight values by adding the squared magnitudes of the weights to the loss function. This encourages smoother decision boundaries, helping the model generalize better by limiting extreme weight values. The decision boundary in this model is more stable and balanced, allowing it to capture important patterns without becoming overly complex.

Logistic Regression with Polynomial Features Decision Boundary: By adding polynomial features like 
𝑥
1
⋅
𝑥
2
x 
1
​
 ⋅x 
2
​
  and 
𝑥
1
2
x 
1
2
​
 , logistic regression gains some capacity to capture non-linear relationships. For the XOR dataset, this results in a decision boundary that is curved but still simpler than that of the MLP models. Logistic regression struggles to capture the XOR pattern completely due to its limited flexibility compared to neural networks, even with additional features.

Performance Metrics:

MLP without Regularization:

Accuracy: 0.9950
Precision: 0.9892
Recall: 1.0000
F1 Score: 0.9946
The unregularized MLP achieves high accuracy and perfect recall, indicating it captures all positive cases with minimal false positives. The absence of regularization allows the model to adapt to complex patterns in the XOR dataset, but it also increases the risk of overfitting, which could reduce generalizability to new data.

MLP with L1 Regularization:

Accuracy: 0.9750
Precision: 0.9888
Recall: 0.9565
F1 Score: 0.9724
With L1 regularization, the MLP sacrifices some accuracy for a simpler model. The high precision indicates the model accurately predicts positive cases with few false positives. However, the drop in recall suggests some positive cases are missed, as the model’s sparsity constraint reduces its ability to capture all details of the XOR dataset, focusing instead on more general patterns.

MLP with L2 Regularization:

Accuracy: 0.9650
Precision: 0.9670
Recall: 0.9565
F1 Score: 0.9617
L2 regularization reduces overfitting by controlling the magnitude of weights, leading to slightly better balanced precision and recall. The smoother decision boundary allows the model to capture the XOR dataset’s structure without excessive complexity. This model is slightly more accurate than the L1 regularized version, showing that it generalizes well by maintaining performance with a stable decision boundary.

Logistic Regression with Polynomial Features:

Accuracy: 0.9500
Precision: 0.9659
Recall: 0.9239
F1 Score: 0.9444
Logistic regression, even with polynomial features, doesn’t match the performance of the MLP models. The added polynomial terms enable it to recognize some non-linear patterns, but it lacks the flexibility of neural networks. Thus, the model struggles to capture the full complexity of the XOR dataset. The decision boundary is quadratic, which explains the relatively lower recall and slightly lower accuracy, as the model is limited in its ability to fully adapt to the dataset’s intricacies.
