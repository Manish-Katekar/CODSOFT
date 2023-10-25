import os 
os.chdir("C:\\Users\\manis\\OneDrive\\Desktop\\python exercises\\project_NLP\\spam_ml")
import joblib

import packages.data_preprocessor as dp
import packages.model_trainer as mt

path_to_data = 'C:\\Users\\manis\\OneDrive\\Desktop\\python exercises\\project_NLP\\spam_ml\\data.csv'


prepared_data = dp.prepare_data(path_to_data,encoding="latin-1")

                                       

# Split the data into training and testing sets with TF-IDF and SMOTE
split_data,tfidf_vectorizer = dp.create_train_test_data_with_smote(prepared_data['text'], prepared_data['label'].values, test_size=0.30, random_state=2023)




model=mt.run_model_training(split_data['x_train'], split_data['x_test'],
                            split_data['y_train'], split_data['y_test'])


joblib.dump(model,'./models/my_spam_model.pkl')

joblib.dump(tfidf_vectorizer, open("./vectors/my_vectorizer.pickel","wb"))