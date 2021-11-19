from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
import nltk
import re
import pickle
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_log_error
import pandas as pd
import numpy as np
import scipy


class Model:
	def __init__(self):
		self.model = Ridge(alpha=1)
		self.Cvectorizer = CountVectorizer(min_df = 15)
		self.Tvectorizer = TfidfVectorizer(max_features = 5000, ngram_range = (1,3),stop_words = "english")
		self.Lbinarizer = LabelBinarizer(sparse_output=True)


	def preprocess(self, df):

		df.brand_name = df.brand_name.fillna('unknown')
		df.item_description = df.item_description.fillna("None")

		print('Final_Nulls:\n',df.isna().mean())

		print('Set_Item_condition_id:',len(set(df.item_condition_id)))
		print('Category_name_set:',len(set(df.category_name)))

		df.item_description = df.item_description + '' + df.name + df.category_name

		#Category name to 3 different info
		sc_1 = []
		sc_2 = []
		sc_3 = []
		for feature in (df['category_name'].values):
		    fs = feature.split('/')
		    a,b,c = fs[0], fs[1], ' '.join(fs[2:])
		    sc_1.append(a)
		    sc_2.append(b)
		    sc_3.append(c)
		df['gender'] = sc_1
		df['item'] = sc_2
		df['type_item'] = sc_3


		ps = PorterStemmer()

		def text_pre_processing(article):
		    corpus = []
		    review = re.sub('[^a-zA-Z0-9]', ' ', article) #Remove everything(Punctuations, Numbers, etc... except the alphabetical words)
		    review = review.lower()                     #lowercasing the words
		    review = review.split()
		    review = [ps.stem(word) for word in review if not word in stopwords.words('english')] #removing stopwords and lemmatizing
		    review = ' '.join(review)
		    corpus.append(review)

		    try: return corpus[0]
		    except: return 0 #Returning final preprocessed article string

		df['item_description'] = df['item_description'].astype(str)
		df['name'] = df['name'].astype(str)

		#df['item_description'] = df['item_description'].apply(lambda row: text_pre_processing(row))
		#df['name'] = df['name'].apply(lambda row: text_pre_processing(row))
		
		def calc_char_len(x): # Calculating the character length of each text data
		    try: return len(x)
		    except: return 0

		def calc_word_len(x): # Calculating the word length of each text data
		    try: return len(x.split(' '))
		    except: return 0

		df['id_char_length'] = df['item_description'].apply(lambda x:calc_char_len(x))
		df['id_word_length'] = df['item_description'].apply(lambda x:calc_word_len(x))

		df['name_length'] = df['name'].apply(lambda x:len(x))
		df['log_id_char_length'] = df['id_char_length'].apply(lambda x:np.log1p(x))

		# creating new feature -> log(1 + character length of item_description)
		df['log_id_word_length'] = df['id_word_length'].apply(lambda x:np.log1p(x))

		return df


	def CountVectorizer_fit(self, df):
		self.Cvectorizer.fit(df["name"]) 
		
	def CountVectorizer_transform(self, df):
		X_name = self.Cvectorizer.transform(df['name'])
		return X_name

	def TFIDFVectorizer_fit(self, df):
		self.Tvectorizer.fit(df['item_description'])

	def TFIDFVectorizer_transform(self, df):
		X_descp = self.Tvectorizer.transform(df["item_description"])
		return X_descp

	def Labelbinarizer_fit(self, df):
		self.Lbinarizer.fit(df['brand_name'])

	def Labelbinarizer_transform(self, df):
		X_brand = self.Lbinarizer.transform(df["brand_name"])
		return X_brand

	def Dummyencoder(self, df, X_name, X_desc, X_brand, dum):
		more_left = pd.concat([pd.get_dummies(df[['item_condition_id','gender','shipping','item', 'type_item']]), pd.DataFrame(columns = (set(dum) - set(pd.get_dummies(df[['item_condition_id','gender','shipping','item', 'type_item']]).columns)) )], axis =1).fillna(0)
		#print(more_left)
		X_dummies = scipy.sparse.csr_matrix(more_left.values)

		X_left = scipy.sparse.csr_matrix(df[['id_char_length', 'id_word_length', 'name_length', 'log_id_char_length', 'log_id_word_length']])

		X = scipy.sparse.hstack((X_dummies, X_desc, X_brand,  X_left, X_name)).tocsr()

		print({'X_dummies_shape':X_dummies.shape, 'X_name_shape':X_name.shape, 'X_desc_shape':X_desc.shape, 'X_brand_shape':X_brand.shape, 'X_left_shape':X_left.shape})
		return X

	def pickle_CountVectorizer(self, path = 'pklmodel/Cvect.pkl'):
		with open(path, 'wb') as f:
			pickle.dump(self.Cvectorizer, f)
			print('PickledCVectorizer at {}'.format(path))

	def pickle_TFIDFVectorizer(self, path = 'pklmodel/TFIDF.pkl'):
		with open(path, 'wb') as f:
			pickle.dump(self.Tvectorizer, f)
			print('PickledTFIDF at {}'.format(path))

	def pickle_Lbinarizer(self, path = 'pklmodel/Lbinarizer.pkl'):
		with open(path, 'wb') as f:
			pickle.dump(self.Lbinarizer, f)
			print('Lbinarizer at {}'.format(path))

	def fit(self, X_train, y):
		self.model.fit(X_train, np.log1p(y))

	def predict(self, X_test):
		preds = self.model.predict(X_test)
		return np.expm1(abs(preds))

	def RMSLE(self, y_test, preds):
		loss = np.sqrt(mean_squared_log_error(y_test, preds))
		return	loss





