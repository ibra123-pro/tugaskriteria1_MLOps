import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def automate_preprocessing(file_path):
    print(f"Memulai Pipeline Otomatisasi untuk: {file_path} \n")

    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Format file tidak didukung!")
        
    print(f"Total Baris: {df.shape[0]}, Total Kolom: {df.shape[1]}")
    
    df_cleaned = df.copy()

    for col in df_cleaned.columns:
        if df_cleaned[col].isnull().sum() > 0:
            if df_cleaned[col].dtype in ['int64', 'float64']:
                median_val = df_cleaned[col].median()
                df_cleaned[col] = df_cleaned[col].fillna(median_val)
            else:
                mode_val = df_cleaned[col].mode()[0]
                df_cleaned[col] = df_cleaned[col].fillna(mode_val)
    print("Penanganan missing values selesai.")

    if df_cleaned.duplicated().sum() > 0:
        initial_rows = df_cleaned.shape[0]
        df_cleaned.drop_duplicates(inplace=True)
        print(f"Berhasil menghapus {initial_rows - df_cleaned.shape[0]} baris duplikat.")
    else:
        print("Tidak ditemukan data duplikat.")

    if 'Tanggal' in df_cleaned.columns:
        df_cleaned['Tanggal'] = pd.to_datetime(df_cleaned['Tanggal'])
        df_cleaned['Tahun'] = df_cleaned['Tanggal'].dt.year
        df_cleaned['Bulan'] = df_cleaned['Tanggal'].dt.month
        df_cleaned['Hari'] = df_cleaned['Tanggal'].dt.day
        print("Fitur 'Tanggal' berhasil dipecah menjadi Tahun, Bulan, dan Hari.")

    if 'Total_Penjualan' in df_cleaned.columns:
        Q1 = df_cleaned['Total_Penjualan'].quantile(0.25)
        Q3 = df_cleaned['Total_Penjualan'].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        lower_bound = Q1 - 1.5 * IQR
        
        df_cleaned['Total_Penjualan'] = np.where(df_cleaned['Total_Penjualan'] > upper_bound, upper_bound, df_cleaned['Total_Penjualan'])
        df_cleaned['Total_Penjualan'] = np.where(df_cleaned['Total_Penjualan'] < lower_bound, lower_bound, df_cleaned['Total_Penjualan'])
        print("Outlier pada 'Total_Penjualan' telah di-capping.")

    if 'Jumlah_Terjual' in df_cleaned.columns:
        bins = [0, 3, 7, float('inf')]
        labels = ['Sedikit', 'Sedang', 'Banyak']
        df_cleaned['Kategori_Volume'] = pd.cut(df_cleaned['Jumlah_Terjual'], bins=bins, labels=labels)
        print("Binning 'Jumlah_Terjual' ke 'Kategori_Volume' selesai.")

    kolom_drop = ['Invoice_ID', 'Nama_Pelanggan', 'Nama_Produk', 'Wilayah', 'Tanggal']
    kolom_drop_ready = [col for col in kolom_drop if col in df_cleaned.columns]
    df_cleaned.drop(columns=kolom_drop_ready, inplace=True)
    
    # Menentukan kolom kategori yang tersisa untuk di-encode
    kolom_kategori = ['Kategori', 'Metode_Pembayaran', 'Channel_Penjualan', 'Kategori_Volume']
    kolom_ready_to_encode = [col for col in kolom_kategori if col in df_cleaned.columns]
    
    if kolom_ready_to_encode:
        df_cleaned = pd.get_dummies(df_cleaned, columns=kolom_ready_to_encode, drop_first=True)
        print(f"One-Hot Encoding selesai untuk {kolom_ready_to_encode}")

    fitur_numerik = ['Harga_Satuan', 'Jumlah_Terjual']
    fitur_numerik_ready = [col for col in fitur_numerik if col in df_cleaned.columns]
    
    if fitur_numerik_ready:
        scaler = StandardScaler()
        df_cleaned[fitur_numerik_ready] = scaler.fit_transform(df_cleaned[fitur_numerik_ready])
        print(f"Standarisasi (StandardScaler) selesai pada {fitur_numerik_ready}")

    print("Pipeline Selesai! Data Siap Dilatih. \n")
    return df_cleaned

if __name__ == "__main__":
    path_dataset = "dataset_penjualan_umkm_pondok_gede_2024_2025.xlsx"
    
    try:
        # Memanggil fungsi otomatisasi
        data_siap_training = automate_preprocessing(path_dataset)

        # Menyimpan hasil olahan menjadi file CSV fisik!
        output_file = "data_penjualan_siap_ml.csv"
        data_siap_training.to_csv(output_file, index=False)
        print(f"[SUKSES] File data bersih berhasil disimpan dengan nama: {output_file}\n")
        
        # Cetak info hasil akhir data untuk memastikan struktur barunya berhasil
        print("--- 5 Baris Pertama Data Hasil Otomatisasi (Siap Masuk Model ML) ---")
        print(data_siap_training.head())
        
    except Exception as e:
        print(f"[ERROR] Terjadi kendala saat menjalankan otomatisasi: {e}")