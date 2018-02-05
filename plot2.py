'''run spark cluster with jupyter notebook'''
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import itertools
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.PuRd):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=8,fontweight='heavy',rotation=90)
    plt.yticks(tick_marks, classes,fontsize=8,fontweight='heavy')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    #for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #    plt.text(j, i, cm[i, j],
    #             horizontalalignment="center",
    #             color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.figure(figsize=(20,20),dpi=1000)
'''Train the model, the code is not included'''

model = nb.fit(df_train)

# apply the model on the test setM
result = model.transform(df_test)

# keep only label and prediction to compute accuracy
predictionAndLabels = result.select("prediction", "label")

# compute accuracy on the test set
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                  metricName="accuracy")

print("Accuracy: {}".format(evaluator.evaluate(predictionAndLabels)))
result.printSchema()
true_label = result.select('label').collect()

predict_label = result.select('prediction').collect()
confusion_matrix_nb = confusion_matrix(np.array(true_label), np.array(predict_label))

confusion_matrix_nb

subject_ordered = sorted(sub_dict,key=sub_dict.get)
label_list=range(0,32)

plot_confusion_matrix(confusion_matrix_nb,classes = label_list,normalize = True, title = 'Confusion matrix of MultinomialNB Model')
