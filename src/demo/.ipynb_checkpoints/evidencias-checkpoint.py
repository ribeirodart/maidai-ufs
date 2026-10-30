"""Descoberta de arquivos de evidência a partir do CSV exportado pelo IPED (pandas + regex).

Formato do IPED (pt-BR): separador ";", colunas
  "Nome";"Atalho";"Tamanho";"Ext";"Marcador";"Categoria";...;"Caminho";"TrackId"
Em inglês as colunas são Name / Ext / Category / Path. O perito pode dar qualquer nome ao CSV.
Sem CSV utilizável, varre a pasta de evidências pela extensão.
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

# Regex aplicado em "Categoria + Ext" (ex.: "Audios ogg", "Other Images jpg", "Outras Imagens jpg")
REGEX_IMAGEM = re.compile(r"imag|jpe?g|png|webp|bmp|gif|heic|tiff?", re.I)
REGEX_AUDIO = re.compile(r"[aá]udio|mp3|wav|ogg|opus|m4a|aac|amr|flac|wma", re.I)

COLUNAS_NOME = ["nome", "name", "filename", "file", "arquivo"]
COLUNAS_CAMINHO = ["caminho", "path", "filepath", "file_path"]
COLUNAS_CATEGORIA = ["categoria", "category", "type", "tipo"]
COLUNAS_EXT = ["ext", "extensão", "extensao", "extension"]

def ler_txt():
    # Procura pelo arquivo de texto da conversa diretamente na pasta de evidências
    caminho_txt = config.OUTPUT_DIR / "text_message.txt"
    
    if not caminho_txt.is_file():
        # Fallback para outros nomes caso o perito tenha nomeado diferente
        caminho_txt = config.OUTPUT_DIR / "conversa_whatsapp.txt"
        
    if not caminho_txt.is_file():
        print("[Llama] Aviso: text_message.txt não encontrado na pasta de evidências. Usando vazio.")
        return ""
        
    try:
        texto = caminho_txt.read_text(encoding="utf-8", errors="replace").strip()
        print(f"[Llama] Histórico de mensagens carregado de {caminho_txt.name} ({len(texto)} caracteres).")
        return texto
    except Exception as e:
        print(f"[Llama] Erro ao ler o arquivo de mensagens: {e}")
        return ""

# Carrega o conteúdo textual do chat
mensagens = ler_txt()


def _achar_coluna(df, candidatos):
    mapa = {str(c).strip().lower(): c for c in df.columns}
    for c in candidatos:
        if c in mapa:
            return mapa[c]
    return None


def ler_csv(path):
    """Lê CSV do IPED (';') ou comum (','), em UTF-8 (com/sem BOM) ou Latin-1."""
    ultimo = None
    for enc in ("utf-8-sig", "latin-1"):
        for sep in (";", ",", None):
            try:
                df = pd.read_csv(path, sep=sep, engine="python", encoding=enc, dtype=str,
                                 keep_default_na=False)
                if df.shape[1] > 1:
                    return df
            except UnicodeDecodeError as e:
                ultimo = e
                break  # troca de encoding
            except Exception as e:  # separador errado etc.
                ultimo = e
    raise ValueError(f"Não foi possível ler o CSV {path}: {ultimo}")


def _resolver(valor, pasta, indice):
    """Acha o arquivo real. O 'Caminho' do IPED é relativo à raiz do caso (ex.: evidence/a1.ogg)."""
    p = Path(valor.replace("\\", "/"))
    for cand in (p, pasta / p, pasta.parent / p, pasta / p.name):
        if cand.is_file():
            return cand
    return indice.get(p.name)


def listar_arquivos(regex, rotulo):
    """Retorna a lista de caminhos cuja Categoria/Ext casa com `regex`."""
    pasta = config.EVIDENCE_DIR
    if not pasta.is_dir():
        raise FileNotFoundError(f"Pasta de evidências não encontrada: {pasta}")

    indice = {p.name: p for p in pasta.rglob("*") if p.is_file()}
    achados = []

    csv = config.FILELIST
    if csv.is_file():
        df = ler_csv(csv)
        col_nome = _achar_coluna(df, COLUNAS_NOME)
        col_cam = _achar_coluna(df, COLUNAS_CAMINHO)
        col_cat = _achar_coluna(df, COLUNAS_CATEGORIA)
        col_ext = _achar_coluna(df, COLUNAS_EXT)
        col_arq = col_cam or col_nome
        if col_arq is None:
            print(f"[{rotulo}] CSV sem coluna de nome/caminho ({list(df.columns)}). Varrendo a pasta.")
        else:
            if col_cat is None and col_ext is None:
                print(f"[{rotulo}] CSV sem Categoria/Ext; usando só a extensão do arquivo.")
            for _, row in df.iterrows():
                valor = str(row[col_arq]).strip()
                if not valor:
                    continue
                real = _resolver(valor, pasta, indice)
                if real is None or not real.is_file():  # ignora pastas e arquivos ausentes
                    continue
                tipo = " ".join([
                    str(row[col_cat]) if col_cat else "",
                    str(row[col_ext]) if col_ext else "",
                    real.suffix,
                ])
                if regex.search(tipo):
                    achados.append(real)
            print(f"[{rotulo}] {csv.name}: {len(achados)} arquivo(s) encontrados.")

    if not achados:
        achados = [p for p in indice.values() if regex.search(p.suffix)]
        print(f"[{rotulo}] Varredura de {pasta}: {len(achados)} arquivo(s) encontrados.")

    return sorted(dict.fromkeys(achados))

def _parse_mensagens(txt):
    """Converte o text_message.txt (colunas separadas por tab) em lista de dicts."""
    msgs = []
    for linha in str(txt or "").splitlines():
        partes = linha.split("\t")
        if len(partes) < 4 or not partes[0].strip().isdigit():
            continue  # cabeçalho ou linha fora do padrão
        msgs.append({"data_hora": partes[1].strip(),
                     "remetente": partes[2].strip(),
                     "mensagem": " ".join(partes[3:]).strip()})
    return msgs
