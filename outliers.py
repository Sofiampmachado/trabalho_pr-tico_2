import pandas as pd


# Função para remover outliers usando o método IQR
def remove_outliers(df):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    return df[~((df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr))).any(axis=1)]


# Caminhos dos arquivos CSV
file_paths = {
    'dehli': 'C:/Users/santo/OneDrive/Desktop/lic.icd/2023-2024/elementos/trabalho pratico 2/normalized_dehli_data.csv',
    'melb': 'C:/Users/santo/OneDrive/Desktop/lic.icd/2023-2024/elementos/trabalho pratico 2/normalized_melb_data.csv',
    'perth': 'C:/Users/santo/OneDrive/Desktop/lic.icd/2023-2024/elementos/trabalho pratico 2/normalized_perth_data.csv'
}

# Processar cada arquivo
for city, path in file_paths.items():
    # Carregar os dados
    data = pd.read_csv(path)

    # Remover outliers
    data_cleaned = remove_outliers(data)

    # Verificar a forma antes e depois da remoção de outliers
    print(f"Forma original para {city}: {data.shape}")
    print(f"Forma após remoção de outliers para {city}: {data_cleaned.shape}")

    # Salvar o DataFrame limpo em um novo arquivo CSV
    output_path = f'C:/Users/santo/OneDrive/Desktop/lic.icd/2023-2024/elementos/trabalho pratico 2/cleaned_{city}_data.csv'
    data_cleaned.to_csv(output_path, index=False)

    print(f"Remoção de outliers concluída para {city} e dados salvos em '{output_path}'.")
