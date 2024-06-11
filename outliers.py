import pandas as pd


# Função para remover outliers usando o método IQR
def remove_outliers(df):
    # Selecionar apenas as colunas numéricas
    df_numeric = df.select_dtypes(include='number')

    # Calcular Q1 (1º quartil) e Q3 (3º quartil)
    q1 = df_numeric.quantile(0.25)
    q3 = df_numeric.quantile(0.75)
    iqr = q3 - q1

    # Filtrar os dados para remover os outliers
    df_filtered = df[~((df_numeric < (q1 - 1.5 * iqr)) | (df_numeric > (q3 + 1.5 * iqr))).any(axis=1)]

    return df_filtered


# Caminhos dos arquivos CSV
file_paths = {
    'dehli': 'normalized_dehli_data.csv',
    'melb': 'normalized_melb_data.csv',
    'perth': 'normalized_perth_data.csv'
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
    output_path = f'cleaned_{city}_data.csv'
    data_cleaned.to_csv(output_path, index=False)

    print(f"Remoção de outliers concluída para {city} e dados salvos em '{output_path}'.")
