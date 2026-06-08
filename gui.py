import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from preprocess import DataPreprocessor
from visualization import DataVisualizer
from model import DecisionTreeModel
from sklearn.model_selection import train_test_split

class ObesityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Obezite Tahmin Modeli Arayüzü")
        self.root.geometry("600x790")
        self.root.resizable(False, False)

        # Temel sınıfların başlatılması
        try:
            self.df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
        except FileNotFoundError:
            messagebox.showerror("Hata", "Veri seti (ObesityDataSet_raw_and_data_sinthetic.csv) bulunamadı!")
            self.root.destroy()
            return
            
        self.visualizer = DataVisualizer()
        self.dt_model = DecisionTreeModel()
        self.preprocessor = DataPreprocessor()
        
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Boy ve Kilo kaldırma seçenekleri için BooleanVar'lar
        self.remove_height = tk.BooleanVar(value=False)
        self.remove_weight = tk.BooleanVar(value=False)

        self.setup_ui()

    def setup_ui(self):
        style = ttk.Style()
        style.configure("TButton", padding=6, font=('Helvetica', 10))
        style.configure("TLabelframe.Label", font=('Helvetica', 11, 'bold'))

        # 1. Dağılım Görüntüleme Butonları
        btn_frame1 = ttk.LabelFrame(self.root, text="1. Veri Keşfi", padding=10)
        btn_frame1.pack(fill="x", padx=15, pady=10)

        ttk.Button(btn_frame1, text="Obezite Sınıflarının Dağılımını Görüntüle", 
                   command=lambda: self.visualizer.plot_class_distribution(self.df)).pack(fill="x", pady=2)
        ttk.Button(btn_frame1, text="Yaş ve Kilo Dağılımlarını Görüntüle", 
                   command=lambda: self.visualizer.plot_age_weight_distribution(self.df)).pack(fill="x", pady=2)

        # 2. Özellik Seçimi (Boy / Kilo kaldırma)
        feature_frame = ttk.LabelFrame(self.root, text="2. Özellik Seçimi (Feature Selection)", padding=10)
        feature_frame.pack(fill="x", padx=15, pady=10)

        ttk.Label(feature_frame, text="Aşağıdaki sütunları veri setinden kaldır:", font=('Helvetica', 10)).pack(anchor="w")

        chk_row = ttk.Frame(feature_frame)
        chk_row.pack(fill="x", pady=4)

        ttk.Checkbutton(
            chk_row,
            text="Boy (Height) sütununu kaldır",
            variable=self.remove_height
        ).pack(side="left", padx=(0, 20))

        ttk.Checkbutton(
            chk_row,
            text="Kilo (Weight) sütununu kaldır",
            variable=self.remove_weight
        ).pack(side="left")

        ttk.Label(
            feature_frame,
            text="Not: Bu seçenekler aktifken model eğitilip Feature Importance grafiğinde değişim gözlemlenebilir.",
            font=('Helvetica', 9, 'italic'),
            foreground='gray'
        ).pack(anchor="w", pady=(4, 0))

        # 3. Test/Eğitim Ayrımı
        split_frame = ttk.LabelFrame(self.root, text="3. Veri Ayırma", padding=10)
        split_frame.pack(fill="x", padx=15, pady=10)

        ttk.Label(split_frame, text="Test Verisi Yüzdesi (Örn: 20):", font=('Helvetica', 10)).pack(side="left")
        self.test_size_entry = ttk.Entry(split_frame, width=10, font=('Helvetica', 10))
        self.test_size_entry.insert(0, "20")
        self.test_size_entry.pack(side="left", padx=10)

        # 4. Model Eğitimi
        train_frame = ttk.LabelFrame(self.root, text="4. Model Eğitimi ve Değerlendirme", padding=10)
        train_frame.pack(fill="x", padx=15, pady=10)

        ttk.Button(train_frame, text="Modeli Eğit", command=self.train_model).pack(fill="x", pady=5)

        # Metrikler
        metric_frame = ttk.Frame(train_frame)
        metric_frame.pack(fill="x", pady=5)
        
        self.lbl_accuracy = ttk.Label(metric_frame, text="Accuracy (Doğruluk): -", font=('Helvetica', 10))
        self.lbl_accuracy.pack(anchor="w")
        
        self.lbl_sensitivity = ttk.Label(metric_frame, text="Sensitivity (Duyarlılık): -", font=('Helvetica', 10))
        self.lbl_sensitivity.pack(anchor="w")

        self.lbl_specificity = ttk.Label(metric_frame, text="Specificity (Özgüllük): -", font=('Helvetica', 10))
        self.lbl_specificity.pack(anchor="w")

        self.lbl_fmeasure = ttk.Label(metric_frame, text="F-measure: -", font=('Helvetica', 10))
        self.lbl_fmeasure.pack(anchor="w")

        # 5. Model Grafikleri
        btn_frame2 = ttk.LabelFrame(self.root, text="5. Model Görselleştirme", padding=10)
        btn_frame2.pack(fill="x", padx=15, pady=10)

        self.btn_cm = ttk.Button(btn_frame2, text="Confusion Matrix Tablosunu Goster", command=self.show_cm, state="disabled")
        self.btn_cm.pack(fill="x", pady=2)

        self.btn_fi = ttk.Button(btn_frame2, text="Feature Importance Tablosunu Goster", command=self.show_fi, state="disabled")
        self.btn_fi.pack(fill="x", pady=2)

        self.btn_tree = ttk.Button(btn_frame2, text="Karar Agacini Goster (Ilk 3 Katman)", command=self.show_tree_gui, state="disabled")
        self.btn_tree.pack(fill="x", pady=2)
        
    def train_model(self):
        try:
            test_size_val = float(self.test_size_entry.get().strip())
            if test_size_val <= 0 or test_size_val >= 100:
                raise ValueError
            test_size = test_size_val / 100.0
        except ValueError:
            messagebox.showerror("Hata", "Lütfen 0 ile 100 arasında geçerli bir yüzde girin (Örn: 20).")
            return

        df_islenmis = self.preprocessor.preprocess_data(self.df)

        # Kullanıcının seçimine göre Height ve/veya Weight sütunlarını kaldır
        cols_to_drop = ['NObeyesdad']
        removed_cols = []
        if self.remove_height.get() and 'Height' in df_islenmis.columns:
            cols_to_drop.append('Height')
            removed_cols.append('Height')
        if self.remove_weight.get() and 'Weight' in df_islenmis.columns:
            cols_to_drop.append('Weight')
            removed_cols.append('Weight')

        if removed_cols:
            print(f"\n[Bilgi] Şu sütunlar veri setinden kaldırıldı: {', '.join(removed_cols)}")

        self.X = df_islenmis.drop(cols_to_drop, axis=1)
        self.y = df_islenmis['NObeyesdad']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=245
        )

        # Eğitim
        # Kullanıcının konsoldaki optimizasyon çıktılarını görebilmesi için optimize_and_train kullanıyoruz.
        self.dt_model.optimize_and_train(self.X_train, self.y_train)

        # Değerlendirme (Grafiği hemen açma, metrikleri arayüze döndür)
        metrics = self.dt_model.evaluate(self.X_test, self.y_test, show_plot=False)

        self.lbl_accuracy.config(text=f"Accuracy (Doğruluk): % {metrics['accuracy']*100:.2f}")
        self.lbl_sensitivity.config(text=f"Sensitivity (Duyarlılık): % {metrics['sensitivity']*100:.2f}")
        self.lbl_specificity.config(text=f"Specificity (Özgüllük): % {metrics['specificity']*100:.2f}")
        self.lbl_fmeasure.config(text=f"F-measure: % {metrics['f1_score']*100:.2f}")

        # Butonları aktifleştir
        self.btn_cm.config(state="normal")
        self.btn_fi.config(state="normal")
        self.btn_tree.config(state="normal")
        
        messagebox.showinfo("Başarılı", "Model başarıyla eğitildi ve metrikler hesaplandı!")

    def show_cm(self):
        # Parametreleri göndererek sadece grafiği çizdirecek şekilde evaluate çağırıyoruz.
        # Hesaplama çok hızlı olduğu için sorun değil.
        self.dt_model.evaluate(self.X_test, self.y_test, show_plot=True)

    def show_fi(self):
        # Sadece grafik gösterimi için Türkçe etiket haritası
        # Model eğitimi self.X.columns (orijinal isimler) ile yapılıyor, bu map'in etkisi yok
        turkce_isimler = {
            'Gender':                        'Cinsiyet',
            'Age':                           'Yaş',
            'Height':                        'Boy (cm)',
            'Weight':                        'Kilo (kg)',
            'family_history_with_overweight':'Ailede Obezite Öyküsü',
            'FAVC':                          'Yüksek Kalorili Gıda Tük.',
            'FCVC':                          'Sebze Tüketim Sıklığı',
            'NCP':                           'Günlük Öğün Sayısı',
            'CAEC':                          'Öğün Arası Atıştırma',
            'SMOKE':                         'Sigara Kullanımı',
            'CH2O':                          'Günlük Su Tüketimi (L)',
            'SCC':                           'Kalori Takibi',
            'FAF':                           'Fiziksel Aktivite Sıklığı',
            'TUE':                           'Teknoloji Kullanım Süresi',
            'CALC':                          'Alkol Tüketim Sıklığı',
            'MTRANS_Bike':                   'Ulaşım: Bisiklet',
            'MTRANS_Motorbike':              'Ulaşım: Motosiklet',
            'MTRANS_Public_Transportation':  'Ulaşım: Toplu Taşıma',
            'MTRANS_Walking':                'Ulaşım: Yürüyüş',
            'Age_Group':                     'Yaş Grubu (0–4)',
            'Caloric_Load':                  'Kalori Yükü (FCVC×NCP)',
            'Sedentary_Index':               'Sedanter Endeks (TUE−FAF)',
            'Genetic_Diet_Risk':             'Genetik+Beslenme Riski',
        }
        gorsel_isimler = [turkce_isimler.get(col, col) for col in self.X.columns]
        self.dt_model.plot_feature_importance(gorsel_isimler)

    def show_tree_gui(self):
        """Egitilmis karar agacini gorsel ve yazisal olarak gosterir."""
        turkce_isimler = {
            'Gender':                        'Cinsiyet',
            'Age':                           'Yas',
            'Height':                        'Boy (cm)',
            'Weight':                        'Kilo (kg)',
            'family_history_with_overweight':'Ailede Obezite Oyku',
            'FAVC':                          'Y.Kalorili Gida',
            'FCVC':                          'Sebze Tuketim Sikligi',
            'NCP':                           'Ogün Sayisi',
            'CAEC':                          'Ogün Arasi Atistirma',
            'SMOKE':                         'Sigara',
            'CH2O':                          'Su Tuketimi (L)',
            'SCC':                           'Kalori Takibi',
            'FAF':                           'Fiziksel Aktivite',
            'TUE':                           'Teknoloji Kullanimi',
            'CALC':                          'Alkol Tuketimi',
            'MTRANS_Bike':                   'Ulasim:Bisiklet',
            'MTRANS_Motorbike':              'Ulasim:Motosiklet',
            'MTRANS_Public_Transportation':  'Ulasim:Toplu Tasima',
            'MTRANS_Walking':                'Ulasim:Yuruyus',
            'Age_Group':                     'Yas Grubu',
        }
        gorsel_isimler = [turkce_isimler.get(col, col) for col in self.X.columns]

        # Obezite siniflarinin Turkce etiketleri (0-6 siralamasiyla)
        sinif_etiketleri = [
            'Yetersiz Kilo',
            'Normal Kilo',
            'Fazla Kilo I',
            'Fazla Kilo II',
            'Obezite Tip I',
            'Obezite Tip II',
            'Obezite Tip III',
        ]
        self.dt_model.show_tree(gorsel_isimler, sinif_etiketleri)

def launch_gui():
    root = tk.Tk()
    app = ObesityApp(root)
    root.mainloop()

if __name__ == "__main__":
    launch_gui()
