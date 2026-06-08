import pandas as pd
import matplotlib
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

class DataPreprocessor:
    def __init__(self):
        self.le = LabelEncoder() # iki seçenekli veriler için label encoding
    
    def preprocess_data(self,df):
        df_processed = df.copy()

        ikili_sutunlar = ["Gender", "family_history_with_overweight", "FAVC", 'SMOKE', 'SCC']

        for col in ikili_sutunlar:
            df_processed[col] = self.le.fit_transform(df[col])

        # çok seçenekli veriler için one-hot encoding
        df_processed = pd.get_dummies(df_processed, columns=['MTRANS'], drop_first=True)#nominal veri

        #ordinal veriler
        df_processed['CALC'] = df_processed['CALC'].map({'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3})
        df_processed['CAEC'] = df_processed['CAEC'].map({'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3})

        # hedef değişken de sayısallaştırılıyor
        # Hedef değişkeni hiyerarşik (Ordinal) olarak sayısallaştırma
        obesity_mapping = {
            'Insufficient_Weight': 0, 
            'Normal_Weight': 1, 
            'Overweight_Level_I': 2, 
            'Overweight_Level_II': 3, 
            'Obesity_Type_I': 4, 
            'Obesity_Type_II': 5, 
            'Obesity_Type_III': 6
        }
        
        df_processed['NObeyesdad'] = df_processed['NObeyesdad'].map(obesity_mapping)

        return df_processed
