import pydotplus
import pandas as pd
from sklearn import tree
from sklearn import metrics


def BuildTree(trainingData, testingData, trainingLabel, testingLabel, featureColumns):
    x_train, y_train = trainingData, trainingLabel
    x_test, y_test = testingData, testingLabel

    min_samples_leaf = [5, 15, 25, 40, 50]
    for leaf in min_samples_leaf:
        clf = tree.DecisionTreeClassifier(min_samples_leaf=leaf).fit(X=x_train, y=y_train)
        y_pred = clf.predict(X=x_test)
        print(pd.DataFrame(metrics.confusion_matrix(y_test, y_pred), columns=['Predicted abnormal', 'Predicted normal'],
                           index=['Abnormal', 'Normal']))
        print(metrics.classification_report(y_true=y_test, y_pred=y_pred))

        dot_data = tree.export_graphviz(decision_tree=clf, out_file=None,
                                        feature_names=featureColumns,
                                        class_names=['Abnormal', 'Normal'],
                                        filled=True, rounded=True,
                                        special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_png('decisionTree_2Classes_Denoted_{}leaves.png'.format(leaf))
        # graph.write_pdf('decisionTree_2Classes_Denoted_{}leaves.pdf'.format(leaf))



