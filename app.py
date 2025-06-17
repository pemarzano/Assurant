import pandas as pd
import numpy as np
from flask import Flask, render_template_string, request, send_file, jsonify, redirect, url_for
from io import BytesIO
from datetime import datetime
from geopy.geocoders import Nominatim
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
import time
import warnings
import os
import logging
from geopy.distance import geodesic
import sys
import io
import json
from pathlib import Path
from sklearn.cluster import DBSCAN, KMeans
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.neighbors import BallTree
import numpy as np


# Configuração para lidar com Unicode no Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('cobertura_oficinas.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
else:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('cobertura_oficinas.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.6f}'.format)

app = Flask(__name__, static_folder='static')
app.secret_key = 'car10analisededados' 


login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Classes de usuário
class User(UserMixin):
    def __init__(self, id):
        self.id = id

# Usuários permitidos (em produção, use um banco de dados)
USERS = {
    'admin': {'password': generate_password_hash('Car10'), 'role': 'admin'},
    'Assurant': {'password': generate_password_hash('Assurant2025'), 'role': 'user'}
}

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# Rota de login (adicionar antes da rota principal)
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username in USERS and check_password_hash(USERS[username]['password'], password):
            user = User(username)
            login_user(user)
            return redirect(url_for('index'))
        
        return render_template_string('''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Login - Heatmap Analyzer</title>
                <style>
                    body {
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        height: 100vh;
                        margin: 0;
                        background-color: #f5f5f5;
                        font-family: Arial, sans-serif;
                    }
                    .login-container {
                        background: white;
                        padding: 30px;
                        border-radius: 10px;
                        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
                        width: 350px;
                        text-align: center;
                    }
                    .login-container h1 {
                        color: #e74c3c;
                        margin-bottom: 20px;
                    }
                    .form-group {
                        margin-bottom: 15px;
                        text-align: left;
                    }
                    .form-group label {
                        display: block;
                        margin-bottom: 5px;
                        font-weight: bold;
                        color: #333;
                    }
                    .form-group input {
                        width: 100%;
                        padding: 10px;
                        border: 1px solid #ddd;
                        border-radius: 5px;
                        box-sizing: border-box;
                    }
                    .login-btn {
                        width: 100%;
                        padding: 12px;
                        background-color: #e74c3c;
                        color: white;
                        border: none;
                        border-radius: 5px;
                        cursor: pointer;
                        font-weight: bold;
                        margin-top: 10px;
                    }
                    .login-btn:hover {
                        background-color: #c0392b;
                    }
                    .error-message {
                        color: #e74c3c;
                        margin-top: 10px;
                    }
                    .logos {
                        display: flex;
                        justify-content: space-around;
                        margin-bottom: 20px;
                    }
                    .logos img {
                        max-height: 50px;
                    }
                </style>
            </head>
            <body>
                <div class="login-container">
                    <div class="logos">
                        <img src="{{ url_for('static', filename='logo.png') }}" alt="Logo">
                        <img src="{{ url_for('static', filename='logo2.png') }}" alt="Logo 2">
                    </div>
                    <h1>Heatmap Analyzer</h1>
                    <form method="post">
                        <div class="form-group">
                            <label for="username">Usuário</label>
                            <input type="text" id="username" name="username" placeholder="Digite seu usuário" required>
                        </div>
                        <div class="form-group">
                            <label for="password">Senha</label>
                            <input type="password" id="password" name="password" placeholder="Digite sua senha" required>
                        </div>
                        <button type="submit" class="login-btn">Entrar</button>
                        <p class="error-message">Credenciais inválidas</p>
                    </form>
                </div>
            </body>
            </html>
        ''')
    
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Login - Heatmap Analyzer</title>
            <style>
                body {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    background-color: #f5f5f5;
                    font-family: Arial, sans-serif;
                }
                .login-container {
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
                    width: 350px;
                    text-align: center;
                }
                .login-container h1 {
                    color: #e74c3c;
                    margin-bottom: 20px;
                }
                .form-group {
                    margin-bottom: 15px;
                    text-align: left;
                }
                .form-group label {
                    display: block;
                    margin-bottom: 5px;
                    font-weight: bold;
                    color: #333;
                }
                .form-group input {
                    width: 100%;
                    padding: 10px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    box-sizing: border-box;
                }
                .login-btn {
                    width: 100%;
                    padding: 12px;
                    background-color: #e74c3c;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                    font-weight: bold;
                    margin-top: 10px;
                }
                .login-btn:hover {
                    background-color: #c0392b;
                }
                .logos {
                    display: flex;
                    justify-content: space-around;
                    margin-bottom: 20px;
                }
                .logos img {
                    max-height: 50px;
                }
            </style>
        </head>
        <body>
            <div class="login-container">
                <div class="logos">
                    <img src="{{ url_for('static', filename='logo.png') }}" alt="Logo">
                    <img src="{{ url_for('static', filename='logo2.png') }}" alt="Logo 2">
                </div>
                <h1>Heatmap Analyzer</h1>
                <form method="post">
                    <div class="form-group">
                        <label for="username">Usuário</label>
                        <input type="text" id="username" name="username" placeholder="Digite seu usuário" required>
                    </div>
                    <div class="form-group">
                        <label for="password">Senha</label>
                        <input type="password" id="password" name="password" placeholder="Digite sua senha" required>
                    </div>
                    <button type="submit" class="login-btn">Entrar</button>
                </form>
            </div>
        </body>
        </html>
    ''')

# ========== FUNÇÕES AUXILIARES ==========
def coordenada_no_brasil(lat, lon):
    """Verifica se as coordenadas estão dentro dos limites aproximados do Brasil"""
    return (lat >= -33.75 and lat <= 5.27 and 
            lon >= -73.99 and lon <= -34.79)

def converter_coordenadas(df):
    df = df.copy()
    
    for col in ['Latitude', 'Longitude']:
        if col not in df.columns:
            raise ValueError(f"Coluna obrigatória '{col}' não encontrada")
    
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    
    df = df.dropna(subset=['Latitude', 'Longitude'])
    valid_coords = df[
        (df['Latitude'].between(-90, 90)) & 
        (df['Longitude'].between(-180, 180)) &
        df.apply(lambda x: coordenada_no_brasil(x['Latitude'], x['Longitude']), axis=1)
    ]
    
    if len(valid_coords) == 0:
        raise ValueError("Nenhuma coordenada válida encontrada após filtragem")
    
    if len(df) != len(valid_coords):
        logging.warning(f"Filtradas {len(df) - len(valid_coords)} coordenadas inválidas")
    
    return valid_coords

def load_cache():
    if CACHE_FILE.exists():
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}        

def save_cidades_cache(cache):
    try:
        # Criar diretório se não existir
        CACHE_CIDADES_FILE.parent.mkdir(exist_ok=True, parents=True)
        with open(CACHE_CIDADES_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"Erro ao salvar cache de cidades: {str(e)}")
        try:
            # Fallback para diretório home do usuário
            alt_path = Path.home() / 'cidades_cache.json'
            with open(alt_path, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            logging.warning(f"Cache de cidades salvo em local alternativo: {alt_path}")
        except Exception as alt_e:
            logging.error(f"Falha ao salvar cache alternativo: {str(alt_e)}")

def limpar_cache_incorreto():
    if CACHE_CIDADES_FILE.exists():
        try:
            with open(CACHE_CIDADES_FILE, 'r', encoding='utf-8') as f:
                cache = json.load(f)
                # Verifica se o cache tem estrutura incorreta
                if any(not isinstance(v, dict) for v in cache.values()):
                    print("⚠️ Cache com estrutura incorreta encontrado - limpando...")
                    CACHE_CIDADES_FILE.unlink()  # Remove o arquivo de cache
                    return {}
        except (json.JSONDecodeError, TypeError):
            print("⚠️ Cache corrompido - limpando...")
            CACHE_CIDADES_FILE.unlink()
            return {}
    return {}            

def load_cidades_cache():
    try:
        if CACHE_CIDADES_FILE.exists():
            with open(CACHE_CIDADES_FILE, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():  # Verifica se o arquivo não está vazio
                    return json.loads(content)
        return {}
    except (json.JSONDecodeError, IOError) as e:
        logging.warning(f"Erro ao carregar cache de cidades: {str(e)} - Criando novo cache")
        return {}

def save_cidades_cache(cache):
    with open(CACHE_CIDADES_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False)

def batch_geocode(coordinates, batch_size=50):
    """Processa coordenadas em lotes para melhor performance"""
    results = {}
    for i in range(0, len(coordinates), batch_size):
        batch = coordinates[i:i+batch_size]
        for lat, lon in batch:
            chave = f"{lat:.4f},{lon:.4f}"
            if chave not in CACHE_CIDADES:
                results[chave] = get_cidade(lat, lon)
        time.sleep(GEOPY_DELAY)  # Delay entre lotes, não entre cada coordenada
    return results   
     
def get_cidade(lat, lon):
    chave = f"{lat:.4f},{lon:.4f}"
    if chave not in CACHE_CIDADES:
        max_retries = 3
        retry_delay = 5  # segundos
        
        for attempt in range(max_retries):
            try:
                geolocator = Nominatim(user_agent=f"cobertura_oficinas_app_{attempt}", timeout=GEOPY_TIMEOUT)
                location = geolocator.reverse((lat, lon), exactly_one=True, language='pt')
                
                if location and location.raw:
                    address = location.raw.get('address', {})
                    cidade_keys = ['city', 'town', 'village', 'municipality', 'county', 'state_district']
                    cidade = next((address.get(key) for key in cidade_keys if key in address), "Não identificado")
                    
                    uf = address.get('state', '')
                    if isinstance(uf, str):
                        # Mapeamento mais completo de estados brasileiros
                        uf_map = {
                            'Acre': 'AC',
                            'Alagoas': 'AL',
                            'Amapá': 'AP',
                            'Amazonas': 'AM',
                            'Bahia': 'BA',
                            'Ceará': 'CE',
                            'Distrito Federal': 'DF',
                            'Espírito Santo': 'ES',
                            'Goiás': 'GO',
                            'Maranhão': 'MA',
                            'Mato Grosso': 'MT',
                            'Mato Grosso do Sul': 'MS',
                            'Minas Gerais': 'MG',
                            'Pará': 'PA',
                            'Paraíba': 'PB',
                            'Paraná': 'PR',
                            'Pernambuco': 'PE',
                            'Piauí': 'PI',
                            'Rio de Janeiro': 'RJ',
                            'Rio Grande do Norte': 'RN',
                            'Rio Grande do Sul': 'RS',
                            'Rondônia': 'RO',
                            'Roraima': 'RR',
                            'Santa Catarina': 'SC',
                            'São Paulo': 'SP',
                            'Sergipe': 'SE',
                            'Tocantins': 'TO'
                        }
                        uf = uf_map.get(uf, uf[-2:] if len(uf) >= 2 else 'ND')
                    
                    CACHE_CIDADES[chave] = {
                        'cidade': cidade,
                        'uf': uf
                    }
                    save_cidades_cache(CACHE_CIDADES)
                    return CACHE_CIDADES[chave]
                else:
                    CACHE_CIDADES[chave] = {
                        'cidade': "Não identificado",
                        'uf': 'ND'
                    }
            except Exception as e:
                if attempt == max_retries - 1:
                    logging.warning(f"Falha ao obter cidade para {lat},{lon} após {max_retries} tentativas: {str(e)}")
                    CACHE_CIDADES[chave] = {
                        'cidade': "Erro na consulta",
                        'uf': 'ND'
                    }
                time.sleep(retry_delay)
                continue
            finally:
                time.sleep(GEOPY_DELAY)
    
    return CACHE_CIDADES[chave]

def validar_e_corrigir_ufs():
    global CIDADES_CACHE
    ufs_validas = {'AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 
                  'MT', 'MS', 'MG', 'PA', 'PB', 'PR', 'PE', 'PI', 'RJ', 'RN', 
                  'RS', 'RO', 'RR', 'SC', 'SP', 'SE', 'TO', 'ND'}
    
    correcoes = 0
    for chave in CIDADES_CACHE:
        if isinstance(CIDADES_CACHE[chave], dict):
            uf = CIDADES_CACHE[chave].get('uf', 'ND')
            if len(uf) != 2 or uf not in ufs_validas:
                # Forçar nova consulta para coordenadas com UFs inválidas
                lat, lon = map(float, chave.split(','))
                CIDADES_CACHE[chave] = get_cidade(lat, lon)
                correcoes += 1
    
    if correcoes > 0:
        save_cidades_cache(CIDADES_CACHE)
        logging.info(f"Corrigidas {correcoes} UFs inválidas no cache")

def calcular_distancia_lote(lote_clientes, oficinas):
    lote_clientes = np.radians(np.asarray(lote_clientes))
    oficinas = np.radians(np.asarray(oficinas))
    
    if lote_clientes.ndim == 1:
        lote_clientes = lote_clientes.reshape(1, -1)
    if oficinas.ndim == 1:
        oficinas = oficinas.reshape(1, -1)
    
    dlat = oficinas[:, 0][:, np.newaxis] - lote_clientes[:, 0]
    dlon = oficinas[:, 1][:, np.newaxis] - lote_clientes[:, 1]
    
    a = np.sin(dlat/2)**2 + np.cos(lote_clientes[:, 0]) * np.cos(oficinas[:, 0][:, np.newaxis]) * np.sin(dlon/2)**2
    return 6371 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def calcular_cobertura_otima(clientes_coords, raio_km):
    if len(clientes_coords) == 0:
        return np.array([]), 0
    
    n_clusters = max(1, min(50, int(len(clientes_coords) / 500))) # Limite de 50 clusters
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    kmeans.fit(clientes_coords)
    centros = kmeans.cluster_centers_
    
    distancias = calcular_distancia_lote(clientes_coords, centros)
    centros_validos = []
    
    for i, centro in enumerate(centros):
        if not coordenada_no_brasil(centro[0], centro[1]):
            continue
        clientes_no_raio = np.sum(distancias[i] <= raio_km * 1.2)  # Buffer de 20%
        if clientes_no_raio > 5:  # Mínimo de 50 clientes
            centros_validos.append(centro)
    
    return np.array(centros_validos), len(centros_validos)

def preprocess_common_radii():
    global CLUSTERS_PREPROCESSED, CIDADES_CACHE
    
    print("⏳ Pré-processando raios comuns (10, 15, 20 km)...")
    start_time = time.time()
    
    for raio in RAIOS_PREPROCESS:
        logging.info(f"Processando raio de {raio}km...")
        
        # Calcular clientes não cobertos
        mascara_nao_cobertos = np.ones(len(coords_clientes), dtype=bool)
        
        for i in range(0, len(coords_clientes), LOTE_CLIENTES):
            lote = coords_clientes[i:i+LOTE_CLIENTES]
            distancias = calcular_distancia_lote(lote, coords_oficinas)
            mascara_nao_cobertos[i:i+len(lote)] = ~np.any(distancias <= raio, axis=0)
        
        clientes_nao_cobertos = coords_clientes[mascara_nao_cobertos]
        logging.info(f"Clientes não cobertos encontrados: {len(clientes_nao_cobertos)}")
        
        centros_info = []
        
        if len(clientes_nao_cobertos) > 0:
            # Usar DBSCAN para identificar áreas densas
            dbscan = DBSCAN(eps=raio/111, min_samples=10).fit(clientes_nao_cobertos)
            clusters = [clientes_nao_cobertos[dbscan.labels_ == i] for i in set(dbscan.labels_) if i != -1]
            
            # Para cada cluster denso, calcular centroid
            centros = [np.mean(cluster, axis=0) for cluster in clusters if len(cluster) > 0]
            
            # Pré-obter cidades apenas para os centróides
            for centro in centros:
                lat, lon = centro[0], centro[1]
                chave = f"{lat:.4f},{lon:.4f}"
                
                if chave not in CIDADES_CACHE:
                    cidade_info = get_cidade(lat, lon)
                    # Garantir que cidade_info é um dicionário
                    if isinstance(cidade_info, str):
                        cidade_info = {'cidade': cidade_info, 'uf': 'ND'}
                    CIDADES_CACHE[chave] = cidade_info
                
                centros_info.append({
                    'lat': lat,
                    'lon': lon,
                    'cidade': CIDADES_CACHE[chave]  # Já verificado que é dicionário
                })
        
        CLUSTERS_PREPROCESSED[raio] = {
            'clientes_nao_cobertos': clientes_nao_cobertos,
            'centros_info': centros_info
        }
    
    save_cidades_cache(CIDADES_CACHE)
    elapsed_time = time.time() - start_time
    print(f"✅ Pré-processamento concluído para raios comuns (tempo: {elapsed_time:.2f}s)")

def preprocessar_sugestoes():
    global CLUSTERS_PREPROCESSED
    
    print("⏳ Pré-processando sugestões para todos os raios...")
    start_time = time.time()
    
    raios_preprocess = [5, 10, 15, 20, 25, 30]  # Raios adicionais
    
    for raio in raios_preprocess:
        logging.info(f"Processando raio de {raio}km...")
        
        # Identificar clientes não cobertos
        mascara_nao_cobertos = np.ones(len(coords_clientes), dtype=bool)
        
        for i in range(0, len(coords_clientes), LOTE_CLIENTES):
            lote = coords_clientes[i:i+LOTE_CLIENTES]
            distancias = calcular_distancia_lote(lote, coords_oficinas)
            mascara_nao_cobertos[i:i+len(lote)] = ~np.any(distancias <= raio, axis=0)
        
        clientes_nao_cobertos = coords_clientes[mascara_nao_cobertos]
        
        # Usar DBSCAN com parâmetros mais sensíveis para identificar áreas densas
        dbscan = DBSCAN(eps=raio/111, min_samples=3).fit(clientes_nao_cobertos)  # Reduzir min_samples
        clusters = [clientes_nao_cobertos[dbscan.labels_ == i] for i in set(dbscan.labels_) if i != -1]
        
        # Para cada cluster denso, calcular centroid
        centros = [np.mean(cluster, axis=0) for cluster in clusters if len(cluster) > 0]
        
        # Pré-obter cidades apenas para os centróides
        centros_info = []
        for centro in centros:
            lat, lon = centro[0], centro[1]
            chave = f"{lat:.4f},{lon:.4f}"
            
            if chave not in CIDADES_CACHE:
                cidade_info = get_cidade(lat, lon)
                # Garantir que cidade_info é um dicionário
                if isinstance(cidade_info, str):
                    cidade_info = {'cidade': cidade_info, 'uf': 'ND'}
                CIDADES_CACHE[chave] = cidade_info
            
            centros_info.append({
                'lat': lat,
                'lon': lon,
                'cidade': CIDADES_CACHE[chave]
            })
        
        CLUSTERS_PREPROCESSED[raio] = {
            'clientes_nao_cobertos': clientes_nao_cobertos,
            'centros_info': centros_info
        }
    
    save_cidades_cache(CIDADES_CACHE)
    elapsed_time = time.time() - start_time
    print(f"✅ Pré-processamento concluído para todos os raios (tempo: {elapsed_time:.2f}s)")  


def preprocess_cidades():
    """Pré-processa cidades para coordenadas conhecidas com progresso"""
    print("⏳ Pré-processando cidades para oficinas...")
    
    # Processar todas as oficinas primeiro
    coords_oficinas = [(row['Latitude'], row['Longitude']) for _, row in oficinas_df.iterrows()]
    
    total = len(coords_oficinas)
    processed = 0
    
    for i in range(0, total, 50):  # Lotes de 50
        batch = coords_oficinas[i:i+50]
        batch_geocode(batch)
        processed += len(batch)
        print(f"Progresso: {processed}/{total} ({processed/total:.1%})")
    
    print("✅ Pré-processamento de cidades concluído")


def preprocess_clusters():
    global CLUSTERS_PREPROCESSED
    raios_preprocess = [5, 10, 15, 20]  # Raios mais comuns
    
    for raio in raios_preprocess:
        # Identificar clientes não cobertos
        mascara_nao_cobertos = np.ones(len(coords_clientes), dtype=bool)
        
        for i in range(0, len(coords_clientes), LOTE_CLIENTES):
            lote = coords_clientes[i:i+LOTE_CLIENTES]
            distancias = calcular_distancia_lote(lote, coords_oficinas)
            mascara_nao_cobertos[i:i+len(lote)] = ~np.any(distancias <= raio, axis=0)
        
        clientes_nao_cobertos = coords_clientes[mascara_nao_cobertos]
        
        # Criar clusters apenas se houver clientes não cobertos
        if len(clientes_nao_cobertos) > 0:
            n_clusters = min(2000, max(1, len(clientes_nao_cobertos) // 10))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(clientes_nao_cobertos)
            
            CLUSTERS_PREPROCESSED[raio] = {
                'clientes_nao_cobertos': clientes_nao_cobertos,
                'kmeans': kmeans
            }
        else:
            CLUSTERS_PREPROCESSED[raio] = {
                'clientes_nao_cobertos': np.array([]),
                'kmeans': None
            }

def sugerir_melhores_locais_greedy(clientes_nao_cobertos, raio, max_sugestoes):
    sugestoes = []
    clientes_restantes = clientes_nao_cobertos.copy()
    
    for _ in range(max_sugestoes):
        if len(clientes_restantes) == 0:
            break
            
        # Encontrar o local que cobre mais clientes restantes
        melhor_local = None
        max_cobertura = 0
        
        # Amostrar possíveis locais (poderia ser os próprios clientes)
        for _ in range(100):  # Número de candidatos a avaliar
            candidato = clientes_restantes[np.random.randint(len(clientes_restantes))]
            distancias = calcular_distancia_lote(clientes_restantes, [candidato])
            cobertura = np.sum(distancias <= raio)
            
            if cobertura > max_cobertura:
                max_cobertura = cobertura
                melhor_local = candidato
        
        if melhor_local is not None:
            sugestoes.append(melhor_local)
            # Remover clientes cobertos
            distancias = calcular_distancia_lote(clientes_restantes, [melhor_local])
            clientes_restantes = clientes_restantes[distancias > raio]
    
    return sugestoes

def batch_geocode(coordinates, batch_size=50):
    """Processa coordenadas em lotes para melhor performance"""
    results = {}
    for i in range(0, len(coordinates), batch_size):
        batch = coordinates[i:i+batch_size]
        for lat, lon in batch:
            chave = f"{lat:.4f},{lon:.4f}"
            if chave not in CACHE_CIDADES:
                try:
                    results[chave] = get_cidade(lat, lon)
                except Exception as e:
                    logging.warning(f"Erro no geocoding para {lat},{lon}: {str(e)}")
                    results[chave] = {'cidade': 'Erro', 'uf': 'ND'}
        time.sleep(GEOPY_DELAY)
    return results



# ========== CONFIGURAÇÕES ==========
RAIOS_PREPROCESS = [10, 15, 20]  # Raios para pré-processar
RAIOS_KM = list(range(5, 101, 5))
CACHE_FILE = Path('geocode_cache.json')
CACHE_CIDADES_FILE = Path('cidades_cache.json')  # Cache específico para cidades
LOTE_CLIENTES = 100000
MAX_CLIENTES_MAP = 5000
GEOPY_TIMEOUT = 10
GEOPY_DELAY = 1
MAX_SUGESTOES = 1000
 
# Inicializar variáveis globais antes de qualquer função que as use
CLUSTERS_PREPROCESSED = {}
coords_clientes = None
coords_oficinas = None
CACHE_CIDADES = load_cache()
CIDADES_CACHE = limpar_cache_incorreto() or load_cidades_cache()

def verificar_e_corrigir_cache():
    global CIDADES_CACHE
    
    cache_corrigido = {}
    for chave, valor in CIDADES_CACHE.items():
        if isinstance(valor, dict):
            cache_corrigido[chave] = valor
        else:
            # Assume que é uma string com o nome da cidade
            cache_corrigido[chave] = {'cidade': valor, 'uf': 'ND'}
    
    if len(cache_corrigido) != len(CIDADES_CACHE):
        CIDADES_CACHE = cache_corrigido
        save_cidades_cache(CIDADES_CACHE)

# ========== CARREGAMENTO DE DADOS ==========
# ========== CARREGAMENTO DE DADOS ==========
print("⏳ Iniciando carregamento de dados...")
try:
    if not os.path.exists('oficinas.xlsx'):
        raise FileNotFoundError("Arquivo 'oficinas.xlsx' não encontrado")
    if not os.path.exists('clientes.xlsx'):
        raise FileNotFoundError("Arquivo 'clientes.xlsx' não encontrado")
    
    oficinas_df_raw = pd.read_excel("oficinas.xlsx", engine='openpyxl')
    clientes_df_raw = pd.read_excel("clientes.xlsx", engine='openpyxl')
    
    oficinas_df = converter_coordenadas(oficinas_df_raw)
    clientes_df = converter_coordenadas(clientes_df_raw)
    
    if len(oficinas_df) == 0 or len(clientes_df) == 0:
        raise ValueError("Dados insuficientes após filtragem")
    
    coords_oficinas = oficinas_df[['Latitude', 'Longitude']].values.astype(np.float32)
    coords_clientes = clientes_df[['Latitude', 'Longitude']].values.astype(np.float32)
    
    oficinas_df.reset_index(drop=True, inplace=True)
    clientes_df.reset_index(drop=True, inplace=True)
    
    print(f"✅ Dados carregados: {len(oficinas_df)} oficinas e {len(clientes_df):,} clientes válidos")
    
    # Pré-processamento completo
    print("⏳ Pré-processando dados...")
    preprocessar_sugestoes()
    print("✅ Pré-processamento concluído")

except Exception as e:
    print(f"❌ Erro crítico ao carregar dados: {str(e)}")
    logging.error(f"❌ Erro crítico ao carregar dados: {str(e)}")
    exit()

# ========== INTERFACE WEB ==========
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Análise de Cobertura - Oficinas</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
    <script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
    <style>
        :root {
            --primary-color: #e74c3c;
            --secondary-color: #c0392b;
            --danger-color: #e74c3c;
            --success-color: #27ae60;
            --warning-color: #f39c12;
            --text-color: #2c3e50;
            --bg-color: #f8f9fa;
        }
        body { 
            margin: 0; 
            padding: 0; 
            font-family: 'Arial', sans-serif; 
            background-color: #f5f5f5;
        }
        #container { 
            display: flex; 
            height: 100vh; 
        }
        #map { 
            flex: 3; 
        }
        #controls { 
            flex: 1; 
            padding: 20px; 
            background: white; 
            overflow-y: auto; 
            box-shadow: -2px 0 10px rgba(0,0,0,0.1);
        }
        .header {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid var(--primary-color);
            width: 100%;
        }
        .logo {
            max-height: 60px;
            margin-right: 15px;
        }
        .title {
            color: var(--primary-color);
            margin: 0;
            font-size: 20px;
            font-weight: bold;
            text-transform: uppercase;
        }
        .slider-container { 
            margin-bottom: 20px; 
        }
        .slider { 
            width: 100%; 
            -webkit-appearance: none;
            height: 8px;
            border-radius: 4px;
            background: #ddd;
            outline: none;
        }
        .slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: var(--primary-color);
            cursor: pointer;
        }
        .value-display { 
            font-weight: bold; 
            color: var(--primary-color); 
            margin: 5px 0 15px;
            font-size: 18px;
        }
        .metric { 
            background: white; 
            padding: 15px; 
            border-radius: 8px; 
            margin-bottom: 20px; 
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-left: 4px solid var(--primary-color);
        }
        .metric h3 {
            color: var(--primary-color);
            margin-top: 0;
            border-bottom: 1px solid #eee;
            padding-bottom: 8px;
        }
        #export-btn, #sugerir-btn, #heatmap-btn, #raio-btn, #exportar-sugestoes-btn {
            color: white; 
            border: none;
            padding: 12px; 
            width: 100%; 
            border-radius: 6px; 
            cursor: pointer;
            font-weight: bold; 
            margin: 10px 0;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        #export-btn {
            background: var(--primary-color);
        }
        #export-btn:hover { 
            background: var(--secondary-color);
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        #sugerir-btn { 
            background: var(--warning-color);
        }
        #sugerir-btn:hover { 
            background: #e67e22;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        #heatmap-btn {
            background: #3498db;
        }
        #heatmap-btn:hover {
            background: #2980b9;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        #raio-btn {
            background: #9b59b6;
        }
        #raio-btn:hover {
            background: #8e44ad;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        #exportar-sugestoes-btn {
            background-color: #f39c12;
        }
        #exportar-sugestoes-btn:hover {
            background-color: #e67e22;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .legend {
            padding: 10px;
            background: white;
            border-radius: 5px;
            box-shadow: 0 1px 5px rgba(0,0,0,0.2);
            font-size: 14px;
        }
        .loading {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.7);
            z-index: 1000;
            justify-content: center;
            align-items: center;
            color: white;
            font-size: 24px;
            flex-direction: column;
        }
        .loading-spinner {
            border: 5px solid #f3f3f3;
            border-top: 5px solid var(--primary-color);
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin-bottom: 15px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .sugestao-info {
            background-color: #fff3cd;
            padding: 12px;
            border-radius: 6px;
            margin-top: 10px;
            border-left: 4px solid var(--warning-color);
            font-size: 14px;
        }
        .popup-content {
            font-size: 14px;
            line-height: 1.5;
        }
        .popup-content b {
            color: var(--primary-color);
        }
        .spacer {
            flex-grow: 1;
        }
    </style>
</head>
<body>
    <div class="loading" id="loading">
        <div class="loading-spinner"></div>
        <div>Processando dados... Por favor aguarde</div>
    </div>
    
    <div id="container">
        <div id="map"></div>
        <div id="controls">
            <div class="header">
                <img src="{{ url_for('static', filename='logo.png') }}" alt="Logo" class="logo">
                <h1 class="title">Heatmap Analyzer</h1>
                <div class="spacer"></div>
                <img src="{{ url_for('static', filename='logo2.png') }}" alt="Logo 2" class="logo">
            </div>
            
            <div class="slider-container">
                <label>Raio de Atendimento (km):</label>
                <input type="range" min="5" max="100" step="5" value="15" id="raioSlider" class="slider">
                <div class="value-display" id="raio-value">15 km</div>
            </div>
            
            <div class="metric">
                <h3>📊 Métricas Atuais</h3>
                <p>👥 <strong>Clientes cobertos:</strong> <span id="cobertos">0</span></p>
                <p>🚫 <strong>Clientes descobertos:</strong> <span id="descobertos" style="color: var(--danger-color);">0</span></p>
                <p>📈 <strong>Percentual coberto:</strong> <span id="percentual">0%</span></p>
                <p>🏭 <strong>Oficinas ativas:</strong> <span id="oficinas-ativas">0</span></p>
            </div>
            
            <button id="sugerir-btn" onclick="sugerirOficinas()">
                🔍 Sugerir Novas Oficinas
            </button>
            
            <div id="sugestao-container" style="display: none;">
                <div class="metric">
                <h3>💡 Sugestão de Cobertura</h3>
               <div class="slider-container">
               <label>Número de Oficinas Sugeridas:</label>
               <input type="range" min="10" max="1000" step="5" value="10" id="numOficinasSlider" class="slider">
               <div class="value-display" id="num-oficinas-value">10</div>
    </div>
              <div class="slider-container">
              <label>Distância Individual por Oficina (km):</label>
              <input type="range" min="1" max="50" step="1" value="5" id="distanciaOficinaSlider" class="slider">
              <div class="value-display" id="distancia-oficina-value">5 km</div>
    </div>
               <p>📌 <strong>Oficinas Processadas:</strong> <span id="oficinas-sugeridas">0</span></p>
               <p>👥 <strong>Total de Beneficiados:</strong> <span id="clientes-adicionais">0</span></p>
               <p>👤 <strong>Clientes Novos cobertos:</strong> <span id="clientes-unicos">0</span></p>
             <div class="sugestao-info">
                 <p>Clique nos marcadores amarelos no mapa para ver as localizações sugeridas</p>
    </div>
</div>
            </div>
            
            <button id="heatmap-btn" onclick="toggleHeatmap()">
                🔥 Alternar Heatmap
            </button>
            
            <button id="raio-btn" onclick="toggleCirculosRaio()">
                🎯 Mostrar Raios de Atendimento
            </button>
            
            <button id="export-btn" onclick="exportarDados()">
                💾 Exportar Relatório Completo
            </button>

            <button id="exportar-sugestoes-btn" onclick="exportarSugestoes()">
                📤 Exportar Sugestões de Oficinas
            </button>
        </div>
    </div>

    <script>
        // Dados para o frontend
        const oficinas = {{ oficinas|tojson }};
        const totalClientes = {{ total_clientes }};
        let currentRaio = 15;
        let sugestoes = [];
        let heatmapLayer = null;
        let circulosRaio = L.layerGroup();
        
        // Mapa
        const map = L.map('map').setView([-15.5, -55], 4.5);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; OpenStreetMap'
        }).addTo(map);
        
        // Legenda
        const legend = L.control({position: 'bottomright'});
        legend.onAdd = function(map) {
            const div = L.DomUtil.create('div', 'legend');
            div.innerHTML = `
                <div><i style="background:#27ae60; width:12px; height:12px; display:inline-block; border-radius:50%;"></i> Oficina</div>
                <div><i style="background:#f39c12; width:12px; height:12px; display:inline-block; border-radius:50%;"></i> Sugestão</div>
                <div><i style="background:#9b59b6; width:12px; height:12px; display:inline-block; border-radius:50%;"></i> Raio de Atendimento</div>
            `;
            return div;
        };
        legend.addTo(map);
        
        // Camadas
        const marcadores = L.layerGroup().addTo(map);
        const marcadores_sugestao = L.layerGroup().addTo(map);
        
        // Mostrar loading
        function showLoading() {
            document.getElementById('loading').style.display = 'flex';
        }
        
        // Esconder loading
        function hideLoading() {
            document.getElementById('loading').style.display = 'none';
        }
        
        // Exportar sugestões
        function exportarSugestoes() {
    showLoading();
    fetch(`/exportar_sugestoes?raio=${currentRaio}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Erro ao exportar sugestões');
            }
            return response.blob();
        })
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `sugestoes_oficinas_top400_${currentRaio}km_${new Date().toISOString().slice(0,10)}.xlsx`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            hideLoading();
        })
        .catch(error => {
            console.error('Erro:', error);
            hideLoading();
            alert('Erro ao exportar sugestões: ' + error.message);
        });
}
let distanciaOficina = 5;


let debounceTimer;
const DEBOUNCE_DELAY = 1000; // 1 segundo

// Modifique o event listener do slider de distância:
document.getElementById('distanciaOficinaSlider').addEventListener('input', function() {
    distanciaOficina = parseInt(this.value);
    document.getElementById('distancia-oficina-value').textContent = `${distanciaOficina} km`;
    
    // Mostrar indicador de processamento
    document.getElementById('distancia-oficina-value').style.color = '#f39c12';
    
    // Limpar timer anterior
    clearTimeout(debounceTimer);
    
    // Configurar novo timer
    debounceTimer = setTimeout(() => {
        if (sugestoes.length > 0) {
            const numOficinas = parseInt(document.getElementById('numOficinasSlider').value);
            filtrarSugestoes(numOficinas);
        }
        // Restaurar cor original
        document.getElementById('distancia-oficina-value').style.color = '';
    }, DEBOUNCE_DELAY);
});


        // Atualizar visualização
        function atualizarVisualizacao(raio) {
            showLoading();
            currentRaio = raio;
            const distanciaIndividual = distanciaOficina;
            document.getElementById('sugestao-container').style.display = 'none';
            marcadores_sugestao.clearLayers();
            marcadores.clearLayers();
            circulosRaio.clearLayers();
            
            document.getElementById('raio-value').textContent = `${raio} km`;
            
            fetch(`/cobertura?raio=${raio}`)
                .then(response => response.json())
                .then(data => {
                    // Adicionar marcadores (todas ativas)
                    data.oficinas_ativas.forEach(oficina => {
                        L.circleMarker(
                            [oficina.lat, oficina.lon], 
                            {
                                radius: 6 + Math.min(Math.log(oficina.clientes)/2, 5),
                                fillColor: '#27ae60',
                                color: '#fff',
                                weight: 1,
                                fillOpacity: 0.8
                            }
                        ).bindPopup(`
                            <div class="popup-content">
                                <b>${oficina.nome || 'Oficina ' + oficina.id}</b><br>
                                📍 ${oficina.lat.toFixed(4)}, ${oficina.lon.toFixed(4)}<br>
                                🔴 <b>Raio:</b> ${raio} km<br>
                                👥 <b>Clientes:</b> ${oficina.clientes.toLocaleString()}
                            </div>
                        `).addTo(marcadores);
                    });
                    
                    // Atualizar métricas
                    document.getElementById('cobertos').textContent = data.clientes_cobertos.toLocaleString();
                    document.getElementById('descobertos').textContent = (totalClientes - data.clientes_cobertos).toLocaleString();
                    document.getElementById('percentual').textContent = `${((data.clientes_cobertos / totalClientes) * 100).toFixed(1)}%`;
                    document.getElementById('oficinas-ativas').textContent = data.oficinas_ativas.length;
                    
                    // Ajustar zoom
                    if (data.oficinas_ativas.length > 0) {
                        const grupo = new L.featureGroup(marcadores.getLayers());
                        map.fitBounds(grupo.getBounds().pad(0.2));
                    }
                    
                    hideLoading();
                })
                .catch(error => {
                    console.error('Erro:', error);
                    hideLoading();
                    alert('Erro ao carregar dados. Por favor tente novamente.');
                });
        }
        
        // Filtrar sugestões com base no número selecionado
      // Na função filtrarSugestoes(), atualize para usar os novos dados:
      
// Event listener para o slider de distância (deve estar fora de qualquer função)
document.getElementById('distanciaOficinaSlider').addEventListener('input', function() {
    distanciaOficina = parseInt(this.value);
    document.getElementById('distancia-oficina-value').textContent = `${distanciaOficina} km`;
    // Não chamar atualizarVisualizacao aqui
});

// Função filtrarSugestoes corrigida
function filtrarSugestoes(numOficinas) {
    if (!sugestoes || sugestoes.length === 0) {
        console.warn("Nenhuma sugestão disponível para filtrar");
        document.getElementById('oficinas-sugeridas').textContent = '0';
        document.getElementById('clientes-adicionais').textContent = '0';
        document.getElementById('clientes-unicos').textContent = '0';
        marcadores_sugestao.clearLayers();
        return;
    }
    
    const sugestoesFiltradas = [...sugestoes].sort((a, b) => b.score - a.score).slice(0, numOficinas);
    
    marcadores_sugestao.clearLayers();
    
    showLoading();
    
    // Calcular totais
    let totalBeneficiados = 0;
    let clientesUnicos = 0;
    
    sugestoesFiltradas.forEach(sugestao => {
        totalBeneficiados += sugestao.clientes_potenciais;
        clientesUnicos += sugestao.clientes_unicos;
    });
    
    // Atualizar marcadores
    sugestoesFiltradas.forEach((sugestao, idx) => {
        L.circleMarker(
            [sugestao.lat, sugestao.lon], 
            {
                radius: 8,
                fillColor: '#f39c12',
                color: '#fff',
                weight: 2,
                fillOpacity: 0.9
            }
        ).bindPopup(`
            <div class="popup-content">
                <b>📍 Sugestão ${idx + 1}</b><br>
                Latitude: ${sugestao.lat.toFixed(4)}<br>
                Longitude: ${sugestao.lon.toFixed(4)}<br>
                🔴 <b>Raio de cobertura:</b> ${currentRaio} km<br>
                👥 <b>Clientes no raio (total):</b> ${sugestao.clientes_potenciais.toLocaleString()}<br>
                👤 <b>Clientes novos cobertos:</b> ${sugestao.clientes_unicos.toLocaleString()}<br>
                 ${sugestao.cidade ? `🏙️ <b>Cidade estimada:</b> ${sugestao.cidade}<br>` : ''}
            </div>
        `).addTo(marcadores_sugestao);
    });
    
    // Atualizar métricas
    document.getElementById('oficinas-sugeridas').textContent = sugestoesFiltradas.length;
    document.getElementById('clientes-adicionais').textContent = totalBeneficiados.toLocaleString();
    document.getElementById('clientes-unicos').textContent = clientesUnicos.toLocaleString();
    
    if (sugestoesFiltradas.length > 0) {
        const grupo = new L.featureGroup(marcadores_sugestao.getLayers());
        map.fitBounds(grupo.getBounds().pad(0.5));
    }
    
    hideLoading();
}

function sugerirOficinas() {
    showLoading();
    const distanciaIndividual = parseInt(document.getElementById('distanciaOficinaSlider').value);
    
    fetch(`/sugerir_oficinas?raio=${currentRaio}&distancia=${distanciaIndividual}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Erro ao obter sugestões');
            }
            return response.json();
        })
        .then(data => {
            if (!data.sugestoes) {
                throw new Error('Dados de sugestões inválidos');
            }
            sugestoes = data.sugestoes || [];
            const numOficinas = parseInt(document.getElementById('numOficinasSlider').value);
            filtrarSugestoes(numOficinas);
            
            document.getElementById('sugestao-container').style.display = 'block';
            hideLoading();
        })
        .catch(error => {
            console.error('Erro:', error);
            hideLoading();
            alert('Erro ao calcular sugestões. Por favor tente novamente.');
            sugestoes = [];
            filtrarSugestoes(0);
        });
}
        
        // Função para calcular clientes únicos cobertos
        function calcularClientesUnicos(sugestoesFiltradas) {
            showLoading();
            return fetch(`/calcular_clientes_unicos?raio=${currentRaio}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    sugestoes: sugestoesFiltradas
                })
            })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                return data.clientes_unicos;
            })
            .catch(error => {
                console.error('Erro ao calcular clientes únicos:', error);
                hideLoading();
                return 0;
            });
        }
        
        // Sugerir novas oficinas
        function sugerirOficinas() {
            showLoading();
            fetch(`/sugerir_oficinas?raio=${currentRaio}`)
                .then(response => response.json())
                .then(data => {
                    sugestoes = data.sugestoes;
                    const numOficinas = parseInt(document.getElementById('numOficinasSlider').value);
                    filtrarSugestoes(numOficinas);
                    
                    document.getElementById('sugestao-container').style.display = 'block';
                    hideLoading();
                })
                .catch(error => {
                    console.error('Erro:', error);
                    hideLoading();
                    alert('Erro ao calcular sugestões. Por favor tente novamente.');
                });
        }
        
        //Clientes Beneficiados
        function calcularClientesBeneficiados(sugestoesFiltradas) {
        showLoading();
         return fetch(`/calcular_clientes_beneficiados?raio=${currentRaio}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            sugestoes: sugestoesFiltradas
        })
    })
         .then(response => response.json())
         .then(data => {
         hideLoading();
         return {
            totalBeneficiados: data.total_beneficiados,
            clientesUnicos: data.clientes_unicos
        };
    })
          .catch(error => {
         console.error('Erro ao calcular clientes:', error);
         hideLoading();
         return {
            totalBeneficiados: 0,
            clientesUnicos: 0
        };
    });
}
        
        // Alternar heatmap
        function toggleHeatmap() {
            if (heatmapLayer) {
                map.removeLayer(heatmapLayer);
                heatmapLayer = null;
            } else {
                showLoading();
                
                fetch('/clientes_heatmap')
                    .then(response => response.json())
                    .then(data => {
                        heatmapLayer = L.heatLayer(data.clientes, {
                            radius: 25,
                            blur: 20,
                            maxZoom: 15,
                            minOpacity: 0.5,
                            gradient: {
                                0.1: 'blue',
                                0.3: 'cyan',
                                0.5: 'lime',
                                0.7: 'yellow',
                                1.0: 'red'
                            }
                        }).addTo(map);
                        
                        hideLoading();
                    })
                    .catch(error => {
                        console.error('Erro ao carregar heatmap:', error);
                        hideLoading();
                        alert('Erro ao carregar dados para heatmap');
                    });
            }
        }
        
        // Alternar círculos de raio
        function toggleCirculosRaio() {
    if (map.hasLayer(circulosRaio)) {
        map.removeLayer(circulosRaio);
        document.getElementById('raio-btn').textContent = '🎯 Mostrar Raios de Atendimento';
    } else {
        showLoading();
        circulosRaio.clearLayers();
        
        // Adiciona círculos para cada oficina ativa
        marcadores.getLayers().forEach(marker => {
            const latlng = marker.getLatLng();
            L.circle(latlng, {
                radius: currentRaio * 1000,
                color: '#9b59b6',
                fillColor: '#9b59b6',
                fillOpacity: 0.1,
                weight: 1
            }).addTo(circulosRaio);
        });
        
        // Adiciona círculos para cada sugestão
        marcadores_sugestao.getLayers().forEach(marker => {
            const latlng = marker.getLatLng();
            L.circle(latlng, {
                radius: currentRaio * 1000,
                color: '#f39c12',
                fillColor: '#f39c12',
                fillOpacity: 0.1,
                weight: 1
            }).addTo(circulosRaio);
        });
        
        circulosRaio.addTo(map);
        document.getElementById('raio-btn').textContent = '❌ Ocultar Raios';
        hideLoading();
    }
        }
        
        // Exportar dados
           function exportarDados() {
          showLoading();
           alert("Solicite a liberação para exportar os dados completos");
             hideLoading();
         }
        
        // Event listeners
        document.getElementById('raioSlider').addEventListener('input', function() {
            atualizarVisualizacao(parseInt(this.value));
        });
        
        document.getElementById('numOficinasSlider').addEventListener('input', function() {
        const numOficinas = Math.round(parseInt(this.value)/5)*5; // Arredonda para o múltiplo de 5 mais próximo
        document.getElementById('num-oficinas-value').textContent = numOficinas;
        filtrarSugestoes(numOficinas);
});
        // Inicialização
        atualizarVisualizacao(15);
    </script>
</body>
</html>
"""

# ========== ROTAS ==========
@app.route('/')
@login_required
def index():
    oficinas_json = [{
        'id': idx + 1,
        'Nome': row.get('Nome', f'Oficina {idx+1}'),
        'Latitude': float(row['Latitude']),
        'Longitude': float(row['Longitude'])
    } for idx, row in oficinas_df.iterrows()]
    
    return render_template_string(
        HTML_TEMPLATE,
        oficinas=oficinas_json,
        total_clientes=len(clientes_df)
    )

@app.route('/cobertura')
@login_required
def obter_cobertura():
    try:
        raio = int(request.args.get('raio'))
        logging.info("Calculando cobertura para raio de %s km", raio)
        clientes_cobertos = 0
        clientes_por_oficina = np.zeros(len(oficinas_df), dtype=np.int32)
        
        # Processar em lotes
        for i in range(0, len(coords_clientes), LOTE_CLIENTES):
            lote = coords_clientes[i:i+LOTE_CLIENTES]
            distancias = calcular_distancia_lote(lote, coords_oficinas)
            
            if distancias.size == 0:
                continue
                
            # Clientes cobertos por pelo menos uma oficina
            cobertura_clientes = np.any(distancias <= raio, axis=0)
            clientes_cobertos += np.sum(cobertura_clientes)
            
            # Clientes por oficina (dentro do raio)
            clientes_por_oficina += np.sum(distancias <= raio, axis=1)
        
        # Preparar resposta
        ativas = []
        for idx, row in oficinas_df.iterrows():
            if idx < len(clientes_por_oficina):
                ativas.append({
                    'id': idx + 1,
                    'nome': row.get('Nome', f'Oficina {idx+1}'),
                    'lat': float(row['Latitude']),
                    'lon': float(row['Longitude']),
                    'clientes': int(clientes_por_oficina[idx])
                })
        
        return jsonify({
            'clientes_cobertos': int(clientes_cobertos),
            'oficinas_ativas': ativas,
            'oficinas_inativas': []
        })
    
    except Exception as e:
        logging.error(f"Erro em /cobertura: {str(e)}", exc_info=True)
        return jsonify({'error': 'Ocorreu um erro', 'details': str(e)}), 500
    
@app.route('/calcular_clientes_beneficiados', methods=['POST'])
@login_required
def calcular_clientes_beneficiados():
    try:
        raio = int(request.args.get('raio'))
        data = request.get_json()
        sugestoes = data.get('sugestoes', [])
        
        if not sugestoes:
            return jsonify({
                'total_beneficiados': 0,
                'clientes_unicos': 0
            })
        
        # Identificar clientes não cobertos pelas oficinas existentes
        mascara_nao_cobertos = np.ones(len(coords_clientes), dtype=bool)
        
        for i in range(0, len(coords_clientes), LOTE_CLIENTES):
            lote = coords_clientes[i:i+LOTE_CLIENTES]
            distancias = calcular_distancia_lote(lote, coords_oficinas)
            mascara_nao_cobertos[i:i+len(lote)] = ~np.any(distancias <= raio, axis=0)
        
        clientes_nao_cobertos = coords_clientes[mascara_nao_cobertos]
        
        if len(clientes_nao_cobertos) == 0:
            return jsonify({
                'total_beneficiados': 0,
                'clientes_unicos': 0
            })
        
        # Calcular total de clientes beneficiados (SOMANDO TODOS, MESMO COM SOBREPOSIÇÃO)
        total_beneficiados = 0
        clientes_cobertos_mask = np.zeros(len(clientes_nao_cobertos), dtype=bool)
        
        for sugestao in sugestoes:
            centro = np.array([[sugestao['lat'], sugestao['lon']]])
            distancias = calcular_distancia_lote(clientes_nao_cobertos, centro)
            no_raio = (distancias <= raio).flatten()
            total_beneficiados += np.sum(no_raio)  # Soma todos, mesmo os já contados
            clientes_cobertos_mask |= no_raio  # Para clientes únicos
        
        return jsonify({
            'total_beneficiados': int(total_beneficiados),  # Soma de todos os clientes em todos os raios
            'clientes_unicos': int(np.sum(clientes_cobertos_mask))  # Clientes únicos cobertos
        })
        
    except Exception as e:
        logging.error(f"Erro em /calcular_clientes_beneficiados: {str(e)}")
        return jsonify({
            'total_beneficiados': 0,
            'clientes_unicos': 0,
            'error': str(e)
        }), 500

@app.route('/sugerir_oficinas')
@login_required
def sugerir_oficinas():
    try:
        raio = int(request.args.get('raio'))
        distancia_minima = int(request.args.get('distancia', 1))  # Reduzir distância mínima
        
        # Usar dados pré-processados se disponíveis
        if raio in CLUSTERS_PREPROCESSED:
            preprocessed = CLUSTERS_PREPROCESSED[raio]
            centros_info = preprocessed.get('centros_info', [])
            clientes_nao_cobertos = preprocessed.get('clientes_nao_cobertos', np.array([]))
        else:
            # Calcular em tempo real se não estiver pré-processado
            mascara_nao_cobertos = np.ones(len(coords_clientes), dtype=bool)
            
            for i in range(0, len(coords_clientes), LOTE_CLIENTES):
                lote = coords_clientes[i:i+LOTE_CLIENTES]
                distancias = calcular_distancia_lote(lote, coords_oficinas)
                mascara_nao_cobertos[i:i+len(lote)] = ~np.any(distancias <= raio, axis=0)
            
            clientes_nao_cobertos = coords_clientes[mascara_nao_cobertos]
            
            # Processar centros em tempo real com parâmetros mais sensíveis
            dbscan = DBSCAN(eps=raio/111, min_samples=3).fit(clientes_nao_cobertos)  # Reduzir min_samples
            clusters = [clientes_nao_cobertos[dbscan.labels_ == i] for i in set(dbscan.labels_) if i != -1]
            centros = [np.mean(cluster, axis=0) for cluster in clusters if len(cluster) > 0]
            
            centros_info = []
            for centro in centros:
                lat, lon = centro[0], centro[1]
                chave = f"{lat:.4f},{lon:.4f}"
                cidade_info = CIDADES_CACHE.get(chave, get_cidade(lat, lon))
                centros_info.append({
                    'lat': lat, 'lon': lon, 'cidade': cidade_info
                })

        # Processar as sugestões
        sugestoes = []
        centros_validos = []
        
        for centro in centros_info:
            lat, lon = centro['lat'], centro['lon']
            
            # Verificar distância mínima entre sugestões (mais tolerante)
            if len(centros_validos) > 0:
                distancias_sugestoes = calcular_distancia_lote(np.array([[lat, lon]]), np.array(centros_validos))
                if np.min(distancias_sugestoes) < distancia_minima:
                    continue
            
            centros_validos.append([lat, lon])
            
            # Calcular métricas
            distancias_novos = calcular_distancia_lote(clientes_nao_cobertos, [[lat, lon]])
            clientes_novos = int(np.sum(distancias_novos <= raio))
            
            if clientes_novos >= 1:  # Reduzir limite mínimo de clientes
                # Amostragem para clientes potenciais
                sample_size = min(5000, len(coords_clientes))
                sample_indices = np.random.choice(len(coords_clientes), sample_size, replace=False)
                distancias_total = calcular_distancia_lote(coords_clientes[sample_indices], [[lat, lon]])
                clientes_potenciais = int(np.sum(distancias_total <= raio) * (len(coords_clientes) / sample_size))
                
                sugestoes.append({
                    'lat': float(lat),
                    'lon': float(lon),
                    'cidade': f"{centro['cidade']['cidade']}/{centro['cidade']['uf']}",
                    'uf': centro['cidade']['uf'],
                    'clientes_potenciais': clientes_potenciais,
                    'clientes_unicos': clientes_novos,
                    'score': float(clientes_novos)
                })

        # Ordenar e limitar - aumentar limite máximo
        sugestoes.sort(key=lambda x: -x['score'])
        sugestoes = sugestoes[:400]  # Manter limite alto
        
        return jsonify({
            'sugestoes': sugestoes,
            'total_beneficiados': sum(s['clientes_potenciais'] for s in sugestoes),
            'clientes_unicos_totais': sum(s['clientes_unicos'] for s in sugestoes)
        })
    
    except Exception as e:
        logging.error(f"Erro em /sugerir_oficinas: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500
    
@app.route('/calcular_clientes_unicos', methods=['POST'])
@login_required
def calcular_clientes_unicos():
    try:
        raio = int(request.args.get('raio'))
        data = request.get_json()
        sugestoes = data.get('sugestoes', [])
        
        if not sugestoes:
            return jsonify({'clientes_unicos': 0})
        
        # Identificar clientes não cobertos pelas oficinas existentes
        mascara_nao_cobertos = np.ones(len(coords_clientes), dtype=bool)
        
        for i in range(0, len(coords_clientes), LOTE_CLIENTES):
            lote = coords_clientes[i:i+LOTE_CLIENTES]
            distancias = calcular_distancia_lote(lote, coords_oficinas)
            mascara_nao_cobertos[i:i+len(lote)] = ~np.any(distancias <= raio, axis=0)
        
        clientes_nao_cobertos = coords_clientes[mascara_nao_cobertos]
        
        if len(clientes_nao_cobertos) == 0:
            return jsonify({'clientes_unicos': 0})
        
        # Verificar quais clientes não cobertos seriam alcançados pelas sugestões
        clientes_cobertos_mask = np.zeros(len(clientes_nao_cobertos), dtype=bool)
        
        for sugestao in sugestoes:
            centro = np.array([[sugestao['lat'], sugestao['lon']]])
            distancias = calcular_distancia_lote(clientes_nao_cobertos, centro)
            clientes_cobertos_mask |= (distancias <= raio).flatten()
        
        return jsonify({
            'clientes_unicos': int(np.sum(clientes_cobertos_mask))
        })
        
    except Exception as e:
        logging.error(f"Erro em /calcular_clientes_unicos: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/clientes_heatmap')
@login_required
def clientes_heatmap():
    try:
        # Amostrar clientes para evitar sobrecarga
        sample_size = min(5000, len(coords_clientes))
        sample_indices = np.random.choice(len(coords_clientes), sample_size, replace=False)
        sample = coords_clientes[sample_indices]
        
        # Converter para formato [lat, lon, intensity] (intensity = 1 para todos)
        heatmap_data = [[float(lat), float(lon), 1] for lat, lon in sample]
        
        return jsonify({
            'clientes': heatmap_data
        })
    except Exception as e:
        logging.error(f"Erro em /clientes_heatmap: {str(e)}")
        return jsonify({'error': str(e)}), 500
        

@app.route('/exportar_sugestoes')
@login_required
def exportar_sugestoes():
    try:
        raio = int(request.args.get('raio'))
        
        # Verificar se temos dados pré-processados COMPLETOS
        if raio in CLUSTERS_PREPROCESSED and 'centros_info' in CLUSTERS_PREPROCESSED[raio]:
            preprocessed = CLUSTERS_PREPROCESSED[raio]
            centros_info = preprocessed.get('centros_info', [])
            clientes_nao_cobertos = preprocessed.get('clientes_nao_cobertos', np.array([]))
        else:
            # Calcular em tempo real se não estiver pré-processado
            mascara_nao_cobertos = np.ones(len(coords_clientes), dtype=bool)
            
            for i in range(0, len(coords_clientes), LOTE_CLIENTES):
                lote = coords_clientes[i:i+LOTE_CLIENTES]
                distancias = calcular_distancia_lote(lote, coords_oficinas)
                mascara_nao_cobertos[i:i+len(lote)] = ~np.any(distancias <= raio, axis=0)
            
            clientes_nao_cobertos = coords_clientes[mascara_nao_cobertos]
            
            # Identificar clusters de clientes não cobertos
            dbscan = DBSCAN(eps=(raio * 1.5)/111, min_samples=5).fit(clientes_nao_cobertos)
            clusters = [clientes_nao_cobertos[dbscan.labels_ == i] for i in set(dbscan.labels_) if i != -1]
            centros = [np.mean(cluster, axis=0) for cluster in clusters if len(cluster) > 0]
            
            centros_info = []
            for centro in centros:
                lat, lon = centro[0], centro[1]
                chave = f"{lat:.4f},{lon:.4f}"
                cidade_info = CIDADES_CACHE.get(chave, get_cidade(lat, lon))
                
                # Garantir que cidade_info é um dicionário
                if isinstance(cidade_info, str):
                    cidade_info = {'cidade': cidade_info, 'uf': 'ND'}
                elif not isinstance(cidade_info, dict):
                    cidade_info = {'cidade': 'Não identificado', 'uf': 'ND'}
                
                centros_info.append({
                    'lat': lat, 'lon': lon, 'cidade': cidade_info
                })

        # Processar as sugestões
        sugestoes = []
        
        # Pré-calcular máscara de clientes não cobertos
        mascara_nao_cobertos = np.ones(len(coords_clientes), dtype=bool)
        for i in range(0, len(coords_clientes), LOTE_CLIENTES):
            lote = coords_clientes[i:i+LOTE_CLIENTES]
            distancias = calcular_distancia_lote(lote, coords_oficinas)
            mascara_nao_cobertos[i:i+len(lote)] = ~np.any(distancias <= raio, axis=0)
        
        clientes_nao_cobertos = coords_clientes[mascara_nao_cobertos]

        for centro in centros_info:
            lat, lon = centro['lat'], centro['lon']
            
            # Garantir que cidade_info é um dicionário
            cidade_info = centro.get('cidade', {})
            if isinstance(cidade_info, str):
                cidade_info = {'cidade': cidade_info, 'uf': 'ND'}
            elif not isinstance(cidade_info, dict):
                cidade_info = {'cidade': 'Não identificado', 'uf': 'ND'}
            
            # Calcular clientes potenciais (todos os clientes no raio)
            distancias_total = calcular_distancia_lote(coords_clientes, [[lat, lon]])
            clientes_potenciais = int(np.sum(distancias_total <= raio))
            
            # Calcular clientes novos (não cobertos por outras oficinas)
            distancias_novos = calcular_distancia_lote(clientes_nao_cobertos, [[lat, lon]])
            clientes_novos = int(np.sum(distancias_novos <= raio))
            
            sugestoes.append({
                'Latitude': lat,
                'Longitude': lon,
                'Cidade': cidade_info.get('cidade', 'Não identificado'),
                'UF': cidade_info.get('uf', 'ND'),
                'Clientes_Potenciais': clientes_potenciais,
                'Clientes_Novos_Cobertos': clientes_novos,
                'Score': clientes_novos
            })
        
        # Criar DataFrame e ordenar
        df_sugestoes = pd.DataFrame(sugestoes)
        if not df_sugestoes.empty:
            df_sugestoes = df_sugestoes.sort_values('Score', ascending=False).head(400)
            # Converter colunas numéricas para garantir o tipo correto
            df_sugestoes['Clientes_Potenciais'] = df_sugestoes['Clientes_Potenciais'].astype(int)
            df_sugestoes['Clientes_Novos_Cobertos'] = df_sugestoes['Clientes_Novos_Cobertos'].astype(int)
            df_sugestoes['Score'] = df_sugestoes['Score'].astype(int)
        
        # Criar arquivo Excel
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Escrever sugestões
            df_sugestoes.to_excel(writer, sheet_name='Sugestoes', index=False)
            
            # Adicionar formatação
            workbook = writer.book
            worksheet = writer.sheets['Sugestoes']
            
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#f39c12',
                'border': 1
            })
            
            for col_num, value in enumerate(df_sugestoes.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            # Adicionar métricas
            metrics_data = {
                'Métrica': ['Total de Sugestões', 'Raio (km)', 'Data da Análise'],
                'Valor': [
                    len(df_sugestoes),
                    raio,
                    datetime.now().strftime('%d/%m/%Y %H:%M')
                ]
            }
            
            if not df_sugestoes.empty:
                metrics_data['Métrica'].extend([
                    'Total Clientes Potenciais',
                    'Total Clientes Novos Cobertos'
                ])
                metrics_data['Valor'].extend([
                    df_sugestoes['Clientes_Potenciais'].sum(),
                    df_sugestoes['Clientes_Novos_Cobertos'].sum()
                ])
            
            pd.DataFrame(metrics_data).to_excel(writer, sheet_name='Métricas', index=False)
        
        output.seek(0)
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f"sugestoes_oficinas_top400_{raio}km_{datetime.now().strftime('%Y%m%d')}.xlsx"
        )
    
    except Exception as e:
        logging.error(f"Erro em /exportar_sugestoes: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500
    
@app.route('/exportar')
@login_required
def exportar():
    try:
        raio = int(request.args.get('raio'))
        clientes_cobertos = 0
        # Garantir que o array tem o mesmo número de elementos que as oficinas
        clientes_por_oficina = np.zeros(len(oficinas_df), dtype=np.int32)
        
        # Processar em lotes menores para evitar problemas de memória
        LOTE = min(LOTE_CLIENTES, 50000)  # Reduzir o tamanho do lote se necessário
        
        for i in range(0, len(coords_clientes), LOTE):
            lote_clientes = coords_clientes[i:i+LOTE]
            
            # Calcular distâncias para todas as oficinas
            distancias = calcular_distancia_lote(lote_clientes, coords_oficinas)
            
            # Clientes cobertos por pelo menos uma oficina
            clientes_cobertos += np.sum(np.any(distancias <= raio, axis=1))
            
            # Clientes por oficina (somar ao longo do eixo 0 - clientes)
            # Garantir que estamos somando apenas as colunas correspondentes às oficinas
            soma_lote = np.sum(distancias <= raio, axis=0)
            
            # Verificar compatibilidade de dimensões antes de somar
            if len(soma_lote) == len(clientes_por_oficina):
                clientes_por_oficina += soma_lote
            else:
                # Se houver incompatibilidade, usar o mínimo de elementos
                min_len = min(len(soma_lote), len(clientes_por_oficina))
                clientes_por_oficina[:min_len] += soma_lote[:min_len]
        
        # Preparar dados para exportação
        df_oficinas = oficinas_df.copy()
        df_oficinas['Clientes_Cobertos'] = clientes_por_oficina
        
        # Adicionar coluna de cidade (com cache)
        print("Obtendo cidades para as oficinas...")
        df_oficinas['Cidade'] = df_oficinas.apply(
            lambda x: get_cidade(x['Latitude'], x['Longitude']), axis=1)
        
        # Ordenar por clientes cobertos
        df_oficinas = df_oficinas.sort_values('Clientes_Cobertos', ascending=False)
        
        # Criar relatório Excel
        print("Criando arquivo Excel...")
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Sheet de oficinas
            df_oficinas.to_excel(writer, sheet_name='Oficinas', index=False)
            
            # Sheet de métricas
            pd.DataFrame({
                'Métrica': ['Total Clientes', 'Clientes Cobertos', '% Cobertura', 
                           'Oficinas Ativas', 'Raio (km)', 'Data da Análise'],
                'Valor': [
                    len(clientes_df),
                    int(clientes_cobertos),
                    f"{(clientes_cobertos/len(clientes_df))*100:.1f}%",
                    len(oficinas_df),
                    raio,
                    datetime.now().strftime('%d/%m/%Y %H:%M')
                ]
            }).to_excel(writer, sheet_name='Métricas', index=False)
        
        output.seek(0)
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f"cobertura_oficinas_{raio}km_{datetime.now().strftime('%Y%m%d')}.xlsx"
        )
    
    except Exception as e:
        logging.error(f"Erro em /exportar: {str(e)}", exc_info=True)
        return jsonify({'error': f"Erro ao exportar: {str(e)}"}), 500
    
@app.route('/limpar_cache', methods=['POST'])
@login_required
def limpar_cache():
    if current_user.id != 'admin':  # Apenas admin pode limpar o cache
        return jsonify({'error': 'Acesso negado'}), 403
    
    global CIDADES_CACHE
    try:
        if CACHE_CIDADES_FILE.exists():
            CACHE_CIDADES_FILE.unlink()
        CIDADES_CACHE = {}
        return jsonify({'status': 'Cache limpo com sucesso'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500    
    
if __name__ == '__main__':
    try:
        port = int(os.environ.get('PORT', 10000))  # Usa PORT se definido, senão 10000
        host = os.environ.get('HOST', '0.0.0.0')
        logging.info("\n🚀 Aplicação pronta! Acesse http://%s:%s", host, port)
        app.run(host=host, port=port, debug=False)
    except Exception as e:
        logging.error("Fatal error starting application: %s", str(e))
        raise