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
	model.CountVectorizer_fit(df)
	X_name = model.CountVectorizer_transform(df)

	print('\nRunning TFIDF\n')
	model.TFIDFVectorizer_fit(df)
	X_desc = model.TFIDFVectorizer_transform(df)

	print('\nRunning LabelBinarizer\n')
	model.Labelbinarizer_fit(df)
	X_brand = model.Labelbinarizer_transform(df)

	print('\nRunning Dummyencoders\n')
	dummy_cols = ['item_condition_id', 'shipping', 'gender_Women',
           'item_Athletic apparel', 'item_Dresses', 'item_Jeans', 'item_Jewelry',
           'item_Shoes', 'item_Sweaters', 'item_Swimwear', 'item_Tops & blouses',
           'item_Underwear', "item_Women's accessories", "item_Women's handbags",
           'type_item_Above knee, mini', 'type_item_Athletic', 'type_item_Blouse',
           'type_item_Boots', 'type_item_Bracelets', 'type_item_Bras',
           'type_item_Cardigan', 'type_item_Earrings',
           'type_item_Fashion sneakers', 'type_item_Hooded',
           'type_item_Knee-length', 'type_item_Necklaces',
           'type_item_Pants, tights, leggings', 'type_item_Sandals',
           'type_item_Shirts & tops', 'type_item_Shorts',
           'type_item_Shoulder Bags', 'type_item_Slim, skinny',
           'type_item_T-shirts', 'type_item_Tank, cami', 'type_item_Tunic',
           'type_item_Two-piece', 'type_item_Wallets']
	X = model.Dummyencoder(df, X_name, X_desc, X_brand, dummy_cols)

	print('\nPreprocessing complete\n')
	
	
	print('TrainShape_X:', X[:len(df_train)].shape)
	print('TrainShape_y:', y.shape)

	print('TestShape:', X[len(df_train):].shape, '\n')

	print('Training model for validation purpose and calculating validation metric(RMSLE)')
	X_train, X_test, y_train, y_test = train_test_split(X[:len(df_train)], y, test_size = 0.25, random_state = 44)
	model.fit(X_train, y_train)
	preds_test = model.predict(X_test)
	print('RMSLE:',model.RMSLE(y_test, preds_test))
	print('RMSLE<0.5:',( model.RMSLE(y_test, preds_test) <0.5), '\n')

	print('Training Model for testing purpose on full train_data')
	model.fit(X[:len(df_train)], y)

	preds = model.predict(X[len(df_train):])
	preds = pd.DataFrame(preds, columns = ['price'])
	submission = pd.concat([df_test, preds], axis = 1 )
	submission = submission[['id', 'price']]

	print('SubmissionFile_Shape:',submission.shape)
	print('Saving submission.csv file')
	submission.to_csv('submission.csv')

	print('Saving trained model to pklmodel/model.pkl')
	with open('pklmodel/model.pkl', 'wb') as f:
		pickle.dump(model, f)

	model.pickle_CountVectorizer()
	model.pickle_TFIDFVectorizer()
	model.pickle_Lbinarizer()
	
	

if __name__ == "__main__":
	build_model()