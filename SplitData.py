def Split(features, classes):
    trainingFeatures = features[0:230]
    testingFeatures = features[230:310]
    trainingClasses = classes[0:230]
    testingClasses = classes[230:310]

    return trainingFeatures, testingFeatures, trainingClasses,  testingClasses
