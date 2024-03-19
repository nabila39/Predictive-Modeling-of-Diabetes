import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score, confusion_matrix, \
    classification_report, auc, roc_curve
import mplcursors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

#read the dataset
file_path = "C:\\Users\\nabil\\Downloads\\db.csv"
data = pd.read_csv(file_path)
#first  step after reading the data from exel sheet to clean it (1 remove the rows contain missing value or
# null , 2 remove outliers )
featuresToClean = ["NPG", "PGL", "DIA", "TSF", "INS", "BMI", "DPF", "AGE", "Diabetic"]
def cleanData_byRemoveOutliers(cleanData, column):
    quantile25 = cleanData[column].quantile(0.25)
    quantile75 = cleanData[column].quantile(0.75)
    iqr = quantile75 - quantile25
    upperBound = quantile75 + (1.5 * iqr)
    lowerBound = quantile25 - (1.5 * iqr)
    colAfter = cleanData.loc[(cleanData[column] < upperBound) & (cleanData[column] > lowerBound)]
    return colAfter

cleanedData = data.copy()
for i in range(len(featuresToClean)):
    coll = featuresToClean[i]
    cleanedData = cleanData_byRemoveOutliers(cleanedData, coll)

#1
pd.set_option('display.max_columns', None)#this for show all coll on the console
summaryStatistics =cleanedData.describe().transpose()
print("Summary statistics of all attributes in the dataset \n")
print(summaryStatistics)
#2
sns.set(style='whitegrid', font_scale=1.2, palette='pastel')
plt.figure(figsize=(6, 6))
sns.countplot(x='Diabetic', data=cleanedData, palette='Set2')
plt.title('Class Label Distribution (Diabetic) ', fontsize=16, color='blue')
plt.xlabel('Diabetic', fontsize=12, color='green')
plt.ylabel('Count', fontsize=12, color='green')
plt.show()
#3
colors = {0: 'gray', 1: 'pink'}
plt.figure(figsize=(12, 6))
sns.histplot(x='AGE', hue='Diabetic', data=cleanedData, bins=12,kde="true", multiple='stack', palette=colors)
plt.title('Diabetics in each age group')
plt.show()
#4
# Set the style
sns.set(style='whitegrid', font_scale=1.2, palette='dark:salmon_r')
plt.figure(figsize=(8, 6))
sns.distplot(cleanedData['AGE'], color='blue', hist_kws={'edgecolor': 'black', 'linewidth': 2}, kde_kws={'linestyle': '--', 'linewidth': 2})
plt.title('Density Plot for Age', fontsize=16, color='darkred')
plt.xlabel('Age', fontsize=14, color='darkblue')
plt.ylabel('Density', fontsize=14, color='darkblue')
plt.show()
#5
sns.set(style='whitegrid', font_scale=1.2, palette='muted')
plt.figure(figsize=(8, 4))
sns.distplot(cleanedData['BMI'], color='red', hist_kws={'edgecolor': 'black', 'linewidth': 2}, kde_kws={'linestyle': '--', 'linewidth': 2})
plt.title('Density Plot for BMI', fontsize=16, color='darkred')
plt.xlabel('BMI', fontsize=14, color='red')
plt.ylabel('Density', fontsize=14, color='red')
plt.show()
#6
sns.set(style='whitegrid', font_scale=1.2)
plt.figure(figsize=(10, 8))
correlation_matrix = cleanedData.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".2f", linewidths=.5)
plt.title('Correlation matrix', fontsize=16, color='navy')
plt.show()
#7
X = cleanedData.drop(["AGE"], axis=1)
y = cleanedData['AGE']
xForTrain, xForTest, yForTrain, yForTest = train_test_split(X, y, test_size=0.2, random_state=86)
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////
#part 2

#1
Xl1 = cleanedData.drop(["AGE", "Diabetic", "INS", "BMI"], axis=1)
yl1 = cleanedData['AGE']
xForTrainL1, xForTest1, yForTrainL1, yForTest1 = train_test_split(Xl1, yl1, test_size=0.2, random_state=86)
modle1 = LinearRegression()
modle1.fit(xForTrainL1, yForTrainL1)
YpredL1 = modle1.predict(xForTest1)
MSE = mean_squared_error(yForTest1, YpredL1)
rSquerdErrorM1 = r2_score(yForTest1, YpredL1)
adjustedRSquared = 1 - (1 - rSquerdErrorM1) * (len(yForTest1) - 1) / (len(yForTest1) - xForTest1.shape[1] - 1)
print("Mean Squared Error For M1:", MSE)
print("R-squared Error :", rSquerdErrorM1)
print("Adjusted R Square :", adjustedRSquared)
interceptLM1 = modle1.intercept_
print("Y-Intercept For M1:", modle1.intercept_)
coefficientsLM1 = modle1.coef_
equationLM1 = f"Age = {interceptLM1:.2f} + "
for i, coff in enumerate(coefficientsLM1):
    equationLM1 += f"{coff:.2f} * {Xl1.columns[i]} + "
equationLM1 = equationLM1[:-2]
print("Equation for Model 1:")
print(equationLM1)
sns.set(style='whitegrid', font_scale=1.2, palette='pastel')
plt.figure(figsize=(8, 6))
sns.scatterplot(x=yForTest1, y=YpredL1, color='green')
plt.xlabel("Actual Age", fontsize=14, color='darkgreen')
plt.ylabel("Predicted Age", fontsize=14, color='darkgreen')
plt.title("Model 1", fontsize=16, color='darkgreen')
annotations = [f"({x:.2f}, {YpredL1[i]:.2f})" for i, x in enumerate(yForTest1)]
cursor = mplcursors.cursor(hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(annotations[sel.target.index]))
plt.show()

#2
mostImportantFeature = correlation_matrix["AGE"].sort_values(ascending=False).index[1] #here we get the most imprtant feture for the target which is age
xForTrainL2 = xForTrain[[mostImportantFeature]]
xForTestL2 = xForTest[[mostImportantFeature]]
modle2 = LinearRegression()
modle2.fit(xForTrainL2, yForTrain)
YpredL2 = modle2.predict(xForTestL2)
MSEm2 = mean_squared_error(yForTest, YpredL2)
rSquerdErrorM2 = r2_score(yForTest, YpredL2)
adjustedRSquared2 = 1 - (1 - rSquerdErrorM2) * (len(yForTest) - 1) / (len(yForTest) - xForTest.shape[1] - 1)
print("Mean Squared Error For Model 2 :", MSEm2)
print("R-squared Error:", rSquerdErrorM2)
print("Adjusted R Square :", adjustedRSquared2)
sns.set(style='whitegrid', font_scale=1.2, palette='deep')
plt.figure(figsize=(8, 6))
sns.scatterplot(x=xForTestL2[mostImportantFeature], y=yForTest, label="Actual", color='blue')
sns.lineplot(x=xForTestL2[mostImportantFeature], y=YpredL2, color='red', label="Predicted", linewidth=2)
plt.xlabel(mostImportantFeature, fontsize=14, color='navy')
plt.ylabel("AGE", fontsize=14, color='navy')
plt.title("Model 2", fontsize=16, color='navy')
plt.legend()
annotations = [f"({x:.2f}, {YpredL2[i]:.2f})" for i, x in enumerate(yForTest)]
cursor = mplcursors.cursor(hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(annotations[sel.target.index]))
plt.show()
print("Y-Intercept For M2:", modle2.intercept_)
equation = f"Age = {modle2.intercept_:.2f}"
for i, coff in enumerate(modle2.coef_):
    equation += f" + {coff:.2f} * {xForTrainL2.columns[i]} "
print("Linear Equation For M2:", equation)
#3
MIF3 = correlation_matrix["AGE"].sort_values(ascending=False).index[1:4]
xForTrainL3 = xForTrain[MIF3]
xForTestL3 = xForTest[MIF3]
modle3 = LinearRegression()
modle3.fit(xForTrainL3, yForTrain)
YpredL3 = modle3.predict(xForTestL3)
MSEm3 = mean_squared_error(yForTest, YpredL3)
rSquerdErrorM3 = r2_score(yForTest, YpredL3)
adjustedRSquared3 = 1 - (1 - rSquerdErrorM3) * (len(yForTest) - 1) / (len(yForTest) - xForTest.shape[1] - 1)
print("Mean Squared Error For Modle 3:", MSEm3)
print("R-squared Error :", rSquerdErrorM3)
print("Adjusted R Square :", adjustedRSquared3)
sns.set(style='whitegrid', font_scale=1.2, palette='pastel')
plt.figure(figsize=(8, 6))
sns.scatterplot(x=yForTest, y=YpredL3, color='purple', marker='o', alpha=0.7)
plt.xlabel("Actual Age", fontsize=14, color='purple')
plt.ylabel("Predicted Age", fontsize=14, color='purple')
plt.title("Model 3", fontsize=16, color='purple')
annotations = [f"({x:.2f}, {YpredL3[i]:.2f})" for i, x in enumerate(yForTest)]
cursor = mplcursors.cursor(hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(annotations[sel.target.index]))
plt.show()
print("Y-Intercept M3:", modle3.intercept_)
equation = f"Age = {modle3.intercept_:.2f}"
for i, coff in enumerate(modle3.coef_):
    equation += f" + {coff:.2f} * {xForTrainL3.columns[i]} "
print("Linear Equation For M3 :", equation)
#part 3
X = cleanedData.drop(["Diabetic"], axis=1)
y = cleanedData['Diabetic']
xForTrain, xForTest1, yForTrain, yForTest1 = train_test_split(X, y, test_size=0.2, random_state=86)
scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(xForTrain)
X_test_scaler = scaler.transform(xForTest1)
plt.figure(figsize=(8, 6))
for i in range(3, 12):
    if i % 2 == 1:
        KNNclasfier = KNeighborsClassifier(n_neighbors=i)
        KNNclasfier.fit(X_train_scaler, yForTrain)
        predected = KNNclasfier.predict_proba(X_test_scaler)[:, 1]
        fpr, tpr, thresholds = roc_curve(yForTest1, predected)
        ROC_AUC = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'k = {i} (AUC = {ROC_AUC:.2f})')
        Ypred_labels = KNNclasfier.predict(X_test_scaler)
        conf_matrix = confusion_matrix(yForTest1, Ypred_labels)
        sensitivity = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
        print(f"\nConfusion Matrix for k = {i}:\n", conf_matrix)
        print(sensitivity)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random', alpha=0.8)
plt.title('ROC Curve for kNN Classification')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()