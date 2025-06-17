#!/bin/bash
# build.sh

echo "Instalando dependências..."
pip install -r requirements.txt

echo "Pré-processando dados..."
python -c "
import pandas as pd
import numpy as np
from app import converter_coordenadas, preprocessar_sugestoes

print('Carregando dados...')
oficinas_df_raw = pd.read_excel('oficinas.xlsx', engine='openpyxl')
clientes_df_raw = pd.read_excel('clientes.xlsx', engine='openpyxl')

print('Convertendo coordenadas...')
oficinas_df = converter_coordenadas(oficinas_df_raw)
clientes_df = converter_coordenadas(clientes_df_raw)

print('Pré-processando sugestões...')
global coords_oficinas, coords_clientes
coords_oficinas = oficinas_df[['Latitude', 'Longitude']].values.astype(np.float32)
coords_clientes = clientes_df[['Latitude', 'Longitude']].values.astype(np.float32)

preprocessar_sugestoes()
print('Pré-processamento concluído!')
"