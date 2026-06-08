import matplotlib.pyplot as plt
import seaborn as sns

class DataVisualizer:

    def __init__(self):
        sns.set_theme(style="whitegrid")

    def plot_class_distribution(self, df, column_name='NObeyesdad'):
        """
        Veri setindeki obezite sınıflarının dağılımını görselleştirir.
        """

        plt.figure(figsize=(10, 6))

        order = df[column_name].value_counts().index
        sns.countplot(y=df[column_name], hue=df[column_name], palette='viridis', order=order, legend=False)
        plt.title("Obezite sınıflandırmalarının dağılımı", fontsize= 14, fontweight='bold')
        plt.xlabel('Kişi sayısı', fontsize=12)
        plt.ylabel('Obezite sınıfı', fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_age_weight_distribution(self, df):
        """
        Veri setindeki bireylerin Yaş ve Kilo dağılımlarını 
        yan yana iki histogram grafiği olarak çizer.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 1. Grafik: Yaş Dağılımı
        sns.histplot(df['Age'], bins=30, kde=True, color='skyblue', ax=axes[0])
        axes[0].set_title('Veri Setindeki Yaş Dağılımı', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Yaş')
        axes[0].set_ylabel('Frekans')

        # 2. Grafik: Kilo Dağılımı
        sns.histplot(df['Weight'], bins=30, kde=True, color='salmon', ax=axes[1])
        axes[1].set_title('Veri Setindeki Kilo Dağılımı', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Kilo (kg)')
        axes[1].set_ylabel('Frekans')

        plt.tight_layout()
        plt.show()
        
