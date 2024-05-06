from scripts.models import SVM_model, DecisionTree_model, LogicalRegression_model, NeuralNetwork_model, KNN_model, NaiveBayes_model, RandomForest_model, AdaBoost_model, GradientBoosting_model, XGBoost_model, LightGBM_model, CatBoost_model
from scripts.create_dataset import load_data_from_file
from sklearn.model_selection import train_test_split
import json

models = [
    #SVM_model(), # SVM model is not included because it takes too long to train
    DecisionTree_model(),
    LogicalRegression_model(),
    NeuralNetwork_model(),
    KNN_model(),
    NaiveBayes_model(),
    RandomForest_model(),
    AdaBoost_model(),
    GradientBoosting_model(),
    XGBoost_model(),
    LightGBM_model(),
    CatBoost_model()
]

path_dataset = r'my_texture_analizer\datasets\texture_training_data.pkl'
(X, Y) = load_data_from_file(path_dataset)

print('Data loaded from file')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, shuffle=True)
print('Data splitted')

print('Start training models')
data = {}
for ind, model in enumerate(models):
    model.fit(X_train, Y_train)
    model.get_report(X_test, Y_test)
    
    data[model.name] = {'report': model.report, 'time_fitting': model.time_fitting, 'time_prediction': model.time_predicting}
    
    model.save(f'my_texture_analizer/models/{model.name}_model')
    print(f'{ind+1}/{len(models)} models trained and saved')
    print('-----------------------------------------------')
    
print('All models trained and saved')

# Save data to a file
with open('models/models_data.json', 'w') as f:
    json.dump(data, f)
print('Data saved to models_data.json')
