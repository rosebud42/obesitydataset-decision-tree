import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree, export_text

from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold

class DecisionTreeModel:
    def __init__(self):
        #karar ağacı modeli tanımlanıyor
        #random state 42 her çalıştığında aynı sonuçları almak için
        self.model = DecisionTreeClassifier(random_state=42)
    
    def train(self, X_train, y_train):
        #model eğitim verisiyle eğitilir
        self.model.fit(X_train, y_train)
    
    def evaluate(self, X_test, y_test, label_encoder=None, show_plot=True):
        y_pred = self.model.predict(X_test)
        
        # karmaşıklık matrisi hesabı
        cm = confusion_matrix(y_test, y_pred)

        # metrik hesaplamaları
        # accuracy (doğruluk)
        accuracy = accuracy_score(y_test, y_pred)
        # sensitivity (duyarlılık)
        sensitivity = recall_score(y_test, y_pred, average='weighted')
        # f1-score (f1-skoru)
        f_measure = f1_score(y_test, y_pred, average='weighted')

        # specificity (özgüllük)
        specificity_list = []
        for i in range(len(cm)):
            tp = cm[i, i]
            fn = np.sum(cm[i,:]) - tp
            fp = np.sum(cm[:,i]) - tp
            tn = np.sum(cm) - (tp + fp + fn)
            specificity = tp / (tp + fp) if (tp + fp) > 0 else 0
            specificity_list.append(specificity)
        specificity = np.mean(specificity_list)


        # evaluation metrikleri konsola yazdırılıyor
        print("\n" + "="*40)
        print("DEĞERLENDİRME SONUÇLARI")
        print("="*40)
        print(f"Accuracy (Doğruluk)   : % {accuracy * 100:.2f}")
        print(f"Sensitivity (Recall)  : % {sensitivity * 100:.2f}")
        print(f"Specificity (Özgüllük): % {specificity * 100:.2f}")
        print(f"F-measure (F1 Score)  : % {f_measure * 100:.2f}")
        print("="*40)

        
        #karmaşıklık matrisi görselleştirme
        if show_plot:
            plt.figure(figsize=(10,8))
            # Eğer etiket isimlerini geri çevirebilirsek grafikte yazıları gösteririz, yoksa sayıları
            labels = label_encoder.classes_ if label_encoder else "auto"
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
            plt.title('Karmaşıklık Matrisi (Confusion Matrix)', fontsize=14, fontweight='bold')
            plt.xlabel('Modelin Tahmin Ettiği Sınıf', fontsize=12)
            plt.ylabel('Gerçek Sınıf', fontsize=12)
            plt.tight_layout()
            plt.show()
            
        return {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'f1_score': f_measure
        }
        
    def optimize_and_train(self, X_train, Y_train):

        cv_stratified = StratifiedKFold(n_splits=5, shuffle=False)
        param_grid = {
            'criterion'         : ['entropy', 'gini'],
            'max_depth'         : [None, 10, 12, 15, 18, 20, 25],
            'min_samples_split' : [2, 3, 4, 5, 8, 12, 20],   # Küçük ve büyük değerlerin ikisi de
            'min_samples_leaf'  : [1, 2, 3, 4, 5, 8],         # aynı şekilde
            'max_features'      : ['sqrt', 'log2', None],
            'class_weight'      : [None, 'balanced'],
        }

        grid_search = GridSearchCV(
            DecisionTreeClassifier(random_state=42),
            param_grid,
            cv=cv_stratified,          # <-- StratifiedKFold
            scoring='accuracy',
            n_jobs=-1,
            refit=True
        )
        grid_search.fit(X_train, Y_train)

        self.model = grid_search.best_estimator_

        print("="*40)
        print("EN İYİ PARAMETRELER")
        print("="*40)
        print(f"En İyi Derinlik (max_depth)     : {grid_search.best_params_['max_depth']}")
        print(f"En İyi Bölünme Min. Örnek      : {grid_search.best_params_['min_samples_split']}")
        print(f"En İyi Yaprak Min. Örnek       : {grid_search.best_params_['min_samples_leaf']}")
        print(f"En İyi Kriter                  : {grid_search.best_params_['criterion']}")
        print(f"En İyi max_features            : {grid_search.best_params_['max_features']}")
        print(f"En İyi class_weight            : {grid_search.best_params_['class_weight']}")
        print(f"En İyi CV Skoru (validation)   : %{grid_search.best_score_*100:.2f}")
        print("="*40)
        
    def plot_feature_importance(self, feature_names):
        """Modelin karar verirken hangi özelliklere ne kadar önem verdiğini hesaplar ve çizer."""
        importances = self.model.feature_importances_
            
        # Özellikleri ve önem derecelerini bir DataFrame'e koyalım
        import pandas as pd
        feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            
        # En önemliden en önemsize doğru sırala
        feature_df = feature_df.sort_values(by='Importance', ascending=False)
            
        # Grafiği Çizdir
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_df, hue='Feature', palette='magma', legend=False)
        plt.title('Karar Ağacı - Özellik Önem Dereceleri (Feature Importance)', fontsize=14, fontweight='bold')
        plt.xlabel('Önem Skoru (0 ile 1 arası)', fontsize=12)
        plt.ylabel('Özellikler', fontsize=12)
        plt.tight_layout()
        plt.show()
            
        # Konsola da yazdıralım ki tam yüzdeleri görelim
        print("\n=== ÖZELLİKLERİN MODELE ETKİSİ (YÜZDE OLARAK) ===")
        for index, row in feature_df.iterrows():
            print(f"{row['Feature']:<35}: % {row['Importance']*100:.2f}")


    def show_tree(self, feature_names, class_names):
        """Eğitilmiş karar ağacını hem yazısal hem de görsel olarak ekrana basar."""
        
        # 1. YAZISAL GÖSTERİM (Terminalde görmek için)
        print("\n=== AĞACIN KURALLARI (İLK 3 KATMAN) ===")
        # Tüm ağacı yazdırırsak binlerce satır olabilir, o yüzden max_depth=3 ile sınırlandırıyoruz
        agac_metni = export_text(self.model, feature_names=list(feature_names), max_depth=3)
        print(agac_metni)

        # 2. GÖRSEL GÖSTERİM 
        print("\nAğacın renkli görseli çizdiriliyor...")
        plt.figure(figsize=(22, 10)) # Grafiğin büyük olması için boyutları artırdık
        
        # plot_tree fonksiyonu ağacı çizer
        plot_tree(self.model, 
                  feature_names=list(feature_names), 
                  class_names=list(class_names), 
                  filled=True,      # Kutuları sınıflara göre renklendirir
                  rounded=True,     # Kutuların köşelerini yuvarlar
                  fontsize=10,      # Yazı boyutu
                  max_depth=3)      # Sadece ilk 3 katmanı çizdir (okunabilirlik için)
                  
        plt.title('Eğitilmiş Karar Ağacının Mantıksal Yapısı (İlk 3 Derinlik)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()