from LogisticRegression import LogisticRegression
import pandas as pd
import json

#Read data
file_path = 'training_data.txt'
df = pd.read_csv(file_path, header=None)

#Feature-Label split
X = df.iloc[:, :-1]
y=df.iloc[:,2]

#Load config
with open('config.json',) as f:
    configs = json.load(f)

#Train model
model = LogisticRegression(alpha=configs['Alpha'], iters=configs['NumIter'], lamb=configs['Lambda'])
model.fit(X, y)

#Predict on training dataset
y_predict=model.predict(X)

#Save model
with open('model.json', 'w') as f:
    json.dump({'theta: ': model.theta.tolist()}, f)

#Save evalution result
with open('classification_report.json', 'w') as f:
    precision, recall, f1_score, accuracy= model.evaluate(y, y_predict)
    result=[]
    for i in range(2):
        result.append(
            {'label: ': i,
            'precision: ': precision[i],
            'recall: ': recall[i],
            'f1_score: ': f1_score[i],
            })
    result.append({'accuracy: ': accuracy})
    json.dump(result, f)

# Predict sample 
sample=[[-0.022742,1.68494], [-0.092742,0.754], [1.44225,-0.5583]]
sample_predict=model.predict(sample)
print('Result predict on sample test (', sample,'):', sample_predict)