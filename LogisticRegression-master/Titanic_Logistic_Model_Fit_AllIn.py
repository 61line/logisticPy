# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
train = pd.read_csv('titanic_train.csv')


# x、y的选取
X = train.iloc[:, [2, 4, 5, 6, 7, 9, 10]]
y = train.iloc[:, 1]



# 相关系数

# 女性、男性  转化为数字
# Encoding categorical data
sex = pd.get_dummies(X['Sex'], prefix = 'Sex')
sex.drop('Sex_male', inplace = True, axis=1)

# 这行代码的作用是将“登船港口”这一分类变量转换为适合模型所需的数字形式，同时避免多重共线性问题。通过这种编码，后续的机器学习模型可以有效地使用这些特征进行训练和预测。
embark = pd.get_dummies(X['Embarked'], prefix = 'Embarked', drop_first=True)


# 这段代码的作用是将“乘客舱级”这一分类变量转换为适合模型所需的数字形式，同时避免多重共线性问题。通过这种编码，后续的机器学习模型可以有效地使用这些特征进行训练和预测。
passenger_class = pd.get_dummies(X['Pclass'], prefix = 'Pclass')
passenger_class.drop('Pclass_3', inplace = True, axis=1)

# 将原始特征集 X 与编码后的虚拟变量（sex、embark 和 passenger_class）合并，形成一个包含所有特征的新 DataFrame。
X.drop(['Sex','Embarked','Pclass'],axis=1,inplace=True)
X = pd.concat([X,sex,embark, passenger_class],axis=1)

#Outliners
# 这行代码的作用是绘制箱形图，帮助识别和分析数据集中可能的异常值，为数据预处理和后续建模提供依据。
sns.boxplot(data= X).set_title("Outlier Box Plot")


# 这行代码的作用是创建一个包含特征和目标变量的新 DataFrame，为后续的线性关系检查和数据分析提供便利。
# 包含了特征与目标变量
linearity_check_df = pd.concat([pd.DataFrame(X),y],axis=1)

# 这段代码的作用是绘制逻辑回归的散点图和回归线，分析不同特征与生存状态之间的关系，帮助理解数据的结构和特征的重要性。
# sns.regplot():  用于绘制散点图和回归线
sns.regplot(x= 'Age', y= 'Survived', data= linearity_check_df, logistic= True).set_title("Log Odds Linear Plot")
sns.regplot(x= 'Fare', y= 'Survived', data= linearity_check_df, logistic= True).set_title("Log Odds Linear Plot")
sns.regplot(x= 'Sex_male', y= 'Survived', data= linearity_check_df, logistic= True).set_title("Log Odds Linear Plot")


# Splitting the dataset into the Training set and Test set
# 这段代码的作用是将数据集划分为训练集和测试集，为后续的模型训练和评估准备数据。
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)





# Feature Scaling #Need to be done after splitting
# 这段代码的作用是对训练集和测试集中的特定特征进行标准化处理，为后续的模型训练和评估做好准备。
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train.iloc[:, [0,3]] = sc.fit_transform(X_train.iloc[:, [0,3]])
X_test.iloc[:, [0,3]] = sc.transform(X_test.iloc[:, [0,3]])





# Fitting Logistic Regression to the Training set
# 这段代码的作用是创建并训练一个！逻辑回归模型！，为后续的预测和评估做好准备。
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)




#Find relevant features
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

# The "accuracy" scoring is proportional to the number of correct
# classifications
# 这段代码的作用是使用递归特征消除与交叉验证的方法来优化逻辑回归模型的特征集，以提高模型的性能和稳定性。
# RFECV 是用于递归特征消除与交叉验证的类。  本算法中特征值不需要二次筛选，因此不需求rfecv
rfecv = RFECV(estimator=classifier, step=1, cv=StratifiedKFold(2), scoring='accuracy')
rfecv.fit(X_train, y_train)
print("Optimal number of features : %d" % rfecv.n_features_)



# Plot number of features VS. cross-validation scores
# 这段代码的作用是绘制特征数量与交叉验证得分之间的关系图，帮助分析和选择最佳的特征集，以提高模型的性能。
# 选择最佳的特征集
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()



# 这段代码的作用是使用 RFE 方法选择最重要的特征，并输出特征的选择情况和排名信息，从而为模型优化提供依据。
from sklearn.feature_selection import RFE

rfe = RFE(classifier, rfecv.n_features_, step=1)
rfe = rfe.fit(X_train, y_train.values.ravel())
print(rfe.support_)
print(rfe.ranking_)


# Can select columns based on the returned mask   可以根据返回的掩码选择列
# X.loc[:, rfe.support_]


# Predicting the Test set results
# 这段代码的作用是使用已训练的逻辑回归模型对测试集进行预测，生成预测结果以便后续评估模型性能。
y_pred = classifier.predict(X_test)





# K-Fold cross validation
# 这段代码的作用是使用 K 折交叉验证评估逻辑回归模型的性能，计算模型的平均准确率和标准差，以便更全面地理解模型的表现。
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
model_accuracy = accuracies.mean()
model_standard_deviation = accuracies.std()
# 这个用于模型的评估和选择





# Making the Confusion Matrix
# 这段代码的作用是生成并展示混淆矩阵，以评估逻辑回归模型在测试集上的预测效果，从而为后续的模型改进和性能分析提供依据。
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)


# 这段代码的作用是生成并输出分类报告，以评估逻辑回归模型在测试集上的分类效果，提供了精确率、召回率和 F1 分数等关键性能指标，有助于全面了解模型的表现。
# 更加全面
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))



#Genarate Reports
import statsmodels.api as sm

#X_set = X[['Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S', 'Pclass_1', 'Pclass_2']]
X_set = X.loc[:, rfe.support_]
X_set = sm.add_constant(X_set)

logit_model=sm.Logit(y,X_set)
result=logit_model.fit()
print(result.summary2())


# GETTING THE ODDS RATIOS, Z-VALUE, AND 95% CI
model_odds = pd.DataFrame(np.exp(result.params), columns= ['OR'])
model_odds['z-value']= result.pvalues
model_odds[['2.5%', '97.5%']] = np.exp(result.conf_int())




#ROC Curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
area_under_curve = roc_auc_score(y_test, classifier.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % area_under_curve)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
#plt.savefig('Log_ROC')
plt.show()