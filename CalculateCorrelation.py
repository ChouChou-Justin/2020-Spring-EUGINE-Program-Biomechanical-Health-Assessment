import numpy as np
import matplotlib.pyplot as plt

def correlation(dataset, featureColumns):
    featureColumns.append('labels')
    corr = round(dataset.corr(), 3)
    array = np.array(corr)
    fig, axes = plt.subplots(figsize=(6, 6))
    axes.imshow(array)
    axes.set_xticks(np.arange(len(featureColumns)))
    axes.set_yticks(np.arange(len(featureColumns)))
    axes.set_xticklabels(featureColumns)
    axes.set_yticklabels(featureColumns)
    axes.set_title("Correlation matrix", fontsize=18)
    plt.setp(axes.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(featureColumns)):
        for j in range(len(featureColumns)):
            axes.text(j, i, array[i, j], ha="center", va="center", color="w")

    fig.tight_layout()
    plt.show()
    fig.savefig('Correlation matrix.png', bbox_inches='tight')
    # fig.savefig('Correlation matrix.pdf', bbox_inches='tight')

    highest_magnitude = 0
    for i in range(len(corr['label'])):
        if (corr['label'][i] != 1) & (corr['label'][i] > highest_magnitude):
            highest_magnitude = corr['label'][i]
            highest_magnitude_label = corr['label'].index[i]

    new_dataset = dataset.drop(columns=[highest_magnitude_label, 'label'])
    featureColumns.remove(highest_magnitude_label)

    featureColumns.remove('labels')
    new_featureColumns = featureColumns

    return new_dataset, new_featureColumns
