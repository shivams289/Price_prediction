
import pickle
import numpy as np
import pandas as pd
#from api.Temp_repo.model import Model
from model import Model

Model = Model()
path = 'pklmodel/model.pkl'
with open(path, 'rb') as f:
    Model.model = pickle.load(f)

path1 = 'pklmodel/Cvect.pkl'
with open(path1, 'rb') as f:
    Model.Cvectorizer = pickle.load(f)

path2 = 'pklmodel/Lbinarizer.pkl'
with open(path2, 'rb') as f:
    Model.Lbinarizer = pickle.load(f)

path3 = 'pklmodel/TFIDF.pkl'
with open(path3, 'rb') as f:
    Model.Tvectorizer = pickle.load(f)



class PredictPrice:
    def __int__(self):
        # use parser and find the user's query
        pass

    def Doeverything(self, dic):
        test = pd.DataFrame(dic, index = [0])
        columns = ['id', 'name', 'item_condition_id', 
                            'category_name', 'brand_name', 'price', 'shipping', 'item_description', 'seller_id']

        cols_missing = list(set(columns) - set(test.columns))
        cols_missing_df = pd.DataFrame(columns = cols_missing)
        test = pd.concat([test, cols_missing_df], axis = 1)
        
        test = Model.preprocess(test)

        print('\nRunning Countvectorizer\n')
        X_name = Model.CountVectorizer_transform(test)

        print('\nRunning TFIDF\n')
        X_desc = Model.TFIDFVectorizer_transform(test)

        print('\nRunning LabelBinarizer\n')
        X_brand = Model.Labelbinarizer_transform(test)

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




        test = Model.Dummyencoder(test, X_name, X_desc, X_brand, dummy_cols)
        print(test.shape, X_name.shape, X_desc.shape, X_brand.shape)
        prediction = Model.predict(test)

        # create JSON object
        output = {'price': prediction[0]}

        return output

if __name__ == "__main__":

    mo = PredictPrice()
    dic = {'id': 17,
 'name': 'Hold Alyssa Frye Harness boots 12R, Sz 7',
 'item_condition_id': 3,
 'category_name': 'Women/Shoes/Boots',
 'brand_name': 'Frye',
 'price': 79,
 'shipping': 1,
 'seller_id': 211140753,
 'item_description': "Good used condition Women's Fyre harness boots. Very light wear on the heel. Some creasing, distress and wear on leather (naturally distressed leather will distress more with every wear). Light wear on toe box. Please ask any questions as I want you to be happy with your purchase."}
    print(mo.Doeverything(dic)) 