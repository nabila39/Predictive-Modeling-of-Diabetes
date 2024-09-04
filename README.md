# Diabetes Prediction Project

## Project Overview
In this individual project, we develop a model to predict whether a person has diabetes based on various health attributes. This involves performing exploratory data analysis (EDA), feature selection, and applying regression and classification algorithms to assess model performance.

## Dataset
The dataset for this project can be accessed at the following link:
[Diabetes Dataset](https://www.dropbox.com/scl/fi/ahlg01iial19mfl7wrjsy/Diabetes.csv?rlkey=7vwl95ly3hcdvqmwo7t3ply4j&dl=0)

### Data Dictionary
- **NPG**: Number of pregnancies
- **PLG**: Plasma glucose concentration
- **DIA**: Diastolic blood pressure
- **TSF**: Triceps skin fold thickness
- **INS**: 2-Hour serum insulin
- **BMI**: Body mass index
- **DPF**: Diabetes pedigree function
- **AGE**: Age
- **Diabetic**: 1 for positive (diabetic) and 0 otherwise

## Tasks

### Exploratory Data Analysis (EDA)
1. **Summary Statistics**: Print the summary statistics of all attributes in the dataset.
2. **Class Label Distribution**: Show the distribution of the class label (Diabetic) and highlight any significant aspects of this distribution.
3. **Age Group Histogram**: For each age group, draw a histogram detailing the number of diabetics in each sub-group.
4. **Density Plot for Age**: Display the density plot for the `Age` attribute.
5. **Density Plot for BMI**: Display the density plot for the `BMI` attribute.
6. **Correlation Analysis**: Visualize the correlation between all features and explain them. Based on the correlation, decide which features to keep for the learning stage and which to remove.
7. **Data Splitting**: Split the dataset into training (80%) and test (20%) sets.

### Regression Tasks
1. **Linear Regression Model (LR1)**: Apply linear regression to predict `Age` using all independent attributes.
2. **Linear Regression Model (LR2)**: Apply linear regression using the most important feature based on the correlation matrix. Explain the choice of this single attribute.
3. **Linear Regression Model (LR3)**: Apply linear regression using the set of the 3 most important features based on the correlation matrix. Explain why these 3 attributes were chosen.
4. **Model Comparison**: Compare the performance of these models using appropriate performance metrics and explain the differences.

### Classification Tasks
1. **k-Nearest Neighbours (kNN) Classifier**: Run a kNN classifier to predict the `Diabetic` feature using the test set.
2. **kNN with Different Values of k**: Generate kNN classifiers using at least 4 different values of k. Compare their performance using appropriate metrics, including ROC/AUC scores and confusion matrices. Report the results in a table and explain why one model outperforms the others.

## Evaluation Metrics
- Use metrics such as accuracy, precision, recall, F1 score, ROC/AUC score, and confusion matrix to evaluate model performance.

## Results and Analysis
- **Regression Model Performance**: Provide a detailed comparison of the regression models and discuss the impact of different features on performance.
- **Classification Model Performance**: Compare the performance of kNN models with different values of k and analyze the reasons behind the performance differences.

## Conclusion
- Summarize the key findings from the regression and classification tasks, and discuss the implications of these results for predicting diabetes.

## References
- Dataset: [Diabetes Dataset](https://www.dropbox.com/scl/fi/ahlg01iial19mfl7wrjsy/Diabetes.csv?rlkey=7vwl95ly3hcdvqmwo7t3ply4j&dl=0)
