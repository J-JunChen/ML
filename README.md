# Machine Learning experiment
## Task1: Text
### Dataset: [20 Newsgroups dataset](http://qwone.com/~jason/20Newsgroups/)
  - Introduction
    - The 20 Newsgroups data set is a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups. 
    - The data is organized into 20 different newsgroups, each corresponding to a different topic. Some of the newsgroups are very closely related to each other (e.g. comp.sys.ibm.pc.hardware / comp.sys.mac.hardware), while others are highly unrelated (e.g misc.forsale / soc.religion.christian).

  - Performance Evalution
    - Mean of Accuracy (MOA): I had tested 10 times for each model.  STD of Accuracy (SOA) : calculate the standard deviation for the 10 times for each model F1-score
    
    Model | Mean of Accuracy | STD of Accuracy | F1-score
    -- | :--:|:--:|--:
    Naive Bayes|0.8234|0|0.82
    SVM|0.1336|0|0.03
    Linear SVM|0.8079|0|0.81
    Logistic Regression|0.7821|0|0.78
    Multilayer Perceptron|0.8188|0.00144|0.82
    KNN|0.1615|0|0.15
    Decision Tree|0.5066|0.00507|0.51
    Random Forest|0.5110|0.01136|0.49
    Adaboost|0.5537|0|0.56

  - Result Analysis
    - For some model, the STD of Accuracy is zero, I change the random state for separate the training set and test set, but it still don’t work.
    - We can compare to different model, Naive Bayes get the best performance; the default SVM is the worst model, but it was replaced to Linear SVM, it can get a good grade. 

  
## Task2: Image
### Dataset: [The Stree View House Numbers (SVHN) Dataset](http://ufldl.stanford.edu/housenumbers/)
  - Introduction
    - SVHN is a real-world image dataset for developing machine learning and object recognition algorithms with minimal requirement on data preprocessing and formatting.
    - It can be seen as similar in flavor to MNIST (e.g., the images are of small cropped digits), but incorporates an order of magnitude more labeled data (over 600,000 digit images) and comes from a significantly harder, unsolved, real world problem (recognizing digits and numbers in natural scene images). 
    - SVHN is obtained from house numbers in Google Street View images. 10 classes, 1 for each digit. Digit '1' has label 1, '9' has label 9 and '0' has label 10. 73257 digits for training, 26032 digits for testing.

  - Performance Evalution
    - Mean of Accuracy (MOA): I had tested 10 times for each model.
    - STD of Accuracy (SOA) : calculate the standard deviation for the 10 times for each model
    - F1-score

    Model | Mean of Accuracy | STD of Accuracy | F1-score
      -- | :--:|:--:|--:
      Naive Bayes|0.1370|0|0.14
      SVM|0.6395|0|0.64
      Linear SVM|0|0|0
      Logistic Regression|0|0|0
      Multilayer Perceptron|0.2085|0.00144|0.07
      KNN|0.3685|0|0.34
      Decision Tree|0.3244|0.00603|0.32
      Random Forest|0.4324|0.01046|0.42
      Adaboost|0.2330|0|0.19

  - Result Analysis
    - Because of the large volume of image dataset, some model can’t classify the whole dataset, and I split the dataset into smaller set. However, my laptop can satisfy the memory for some model, such Linear SVM, so I give up to calculate the metrics of these model, so I set zero to the sheet.
    - Similar to the task1,  for some model, the STD of Accuracy is zero.

