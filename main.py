import OpenFile
import SplitData
import decisionTree_2Classes
import decisionTree_3Classes
import decisionTree_2Classes_Denoted
import CalculateCorrelation

def main():
    # 2Classes
    features, classes, featureColumns = OpenFile.openFile(path='Biomechanical_Data_2Classes.csv')
    trainingFeatures, testingFeatures, trainingClasses,  testingClasses = SplitData.Split(features=features,
                                                                                          classes=classes)
    decisionTree_2Classes.BuildTree(trainingData=trainingFeatures,
                                    testingData=testingFeatures,
                                    trainingLabel=trainingClasses,
                                    testingLabel=testingClasses,
                                    featureColumns=featureColumns)

    # 3Classes
    features, classes, featureColumns = OpenFile.openFile(path='Biomechanical_Data_3Classes.csv')
    trainingFeatures, testingFeatures, trainingClasses, testingClasses = SplitData.Split(features=features,
                                                                                         classes=classes)
    decisionTree_3Classes.BuildTree(trainingData=trainingFeatures,
                                    testingData=testingFeatures,
                                    trainingLabel=trainingClasses,
                                    testingLabel=testingClasses,
                                    featureColumns=featureColumns)

    # 2Classes_denote
    dataset, labels, featureColumns = OpenFile.DenoteFile(path='Biomechanical_Data_2Classes.csv')
    new_dataset, new_featureColumns = CalculateCorrelation.correlation(dataset=dataset, featureColumns=featureColumns)

    trainingFeatures, testingFeatures, trainingClasses, testingClasses = SplitData.Split(features=new_dataset,
                                                                                         classes=labels)
    decisionTree_2Classes_Denoted.BuildTree(trainingData=trainingFeatures,
                                            testingData=testingFeatures,
                                            trainingLabel=trainingClasses,
                                            testingLabel=testingClasses,
                                            featureColumns=new_featureColumns)


main()
