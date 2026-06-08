from preprocess import DataPreprocessor
from visualization import DataVisualizer
import pandas as pd
from sklearn.model_selection import train_test_split
from model import DecisionTreeModel

def main():
    import gui
    print("Obezite Tahmin Modeli Arayüzü Başlatılıyor...")
    gui.launch_gui()
    


if __name__ == "__main__":
    main()
    