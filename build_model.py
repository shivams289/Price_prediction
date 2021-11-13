import pandas as pd
import numpy as np
from model import Model
import pickle
from sklearn.model_selection import train_test_split

def build_model():
	model = Model()

	df_train = pd.read_csv('data/mercari_train.csv')
	df_test = pd.read_csv('data/mercari_test.csv')

	print('Train_Shape:',df_train.shape)
	print('Test_Shape:',df_test.shape, '\n')
	print('---------------------', '\n')
	print('Train_columns:',df_train.columns)
	print('Test_columns:',df_test.columns, '\n')
	print('---------------------', '\n')
	print('Train_Dtypes:\n',df_train.dtypes, '\n')
	print('Test_Dtypes:\n',df_test.dtypes, '\n')
	print('---------------------', '\n')
	print('Train_Nulls:\n',df_train.isna().mean(), '\n')
	print('Test_Nulls:\n',df_test.isna().mean())

	y = df_train['price']

	df_train = df_train.drop(['price'], axis =1)
	df = pd.concat([df_train, df_test], axis = 0)

	df.reset_index(drop = True, inplace = True)

	print('\nRunning preprocess\n')
	df = model.preprocess(df)
	print('\nRunning Countvectorizer\n')
	X_name = model.CountVectorizer(df)
	print('\nRunning TFIDF\n')
	X_desc = model.TFIDFVectorizer(df)
	print('\nRunning LabelBinarizer\n')
	X_brand = model.Labelbinarizer(df)
	print('\nRunning Dummyencoders\n')
	X = model.Dummyencoder(df, X_name, X_desc, X_brand)

	print('\nPreprocessing complete\n')
	
	#X_train, X_test, y_train, y_test = train_test_split(X, y)
	print('TrainShape_X:', X[:len(df_train)].shape)
	print('TrainShape_y:', y.shape)

	print('TestShape:', X[len(df_train):].shape)

	print('Training Model')
	model.fit(X[:len(df_train)], y)

	preds = model.predict(X[len(df_train):])
	preds = pd.DataFrame(preds, columns = ['price'])
	submission = pd.concat([df_test, preds], axis = 1 )
	submission = submission[['id', 'price']]
	print('Saving submission.csv file')
	submission.to_csv('submission.csv')
	print('Saving trained model to model/model.pkl')
	with open('model/model.pkl', 'wb') as f:
		pickle.dump(model, f)
	
	#print('RMSLE:',model.RMSLE(y_test, preds))

if __name__ == "__main__":
	build_model()