"""Leitura e normalização das evidências exportadas pelo IPED ou extrações brutas."""

import json
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

PADRAO_NOME_CHAT = re.compile(r"message|chat|msg|whatsapp", re.I)
REGEX_IMAGEM = re.compile(r"imag|jpe?g|png|webp|bmp|gif|heic|tiff?", re.I)
REGEX_AUDIO = re.compile(r"[aá]udio|mp3|wav|ogg|opus|m4a|aac|amr|flac|wma", re.I)

COLUNAS_NOME = ("nome", "name", "filename", "file", "arquivo")
COLUNAS_CAMINHO = ("caminho", "path", "filepath", "file_path")
COLUNAS_CATEGORIA = ("categoria", "category", "type", "tipo")
COLUNAS_EXT = ("ext", "extensão", "extensao", "extension")


def carregar_mensagens():
    """Carrega mensagens de bancos SQLite (Android/iOS) ou de arquivos JSON/TXT."""
    pasta = config.EVIDENCE_DIR
    if not pasta.is_dir():
        print(f"[Evidências] Pasta não encontrada: {pasta}")
        return []

    mensagens = []
    arquivos = sorted(arquivo for arquivo in pasta.iterdir() if arquivo.is_file())

    # 1. Prioridade: Bancos de dados SQLite
    for arquivo in arquivos:
        nome = arquivo.name.lower()
        if "msgstore" in nome or "wa.db" in nome:
            print(f"[Evidências] Lendo banco Android: {arquivo.name}")
            mensagens.extend(ler_sqlite_android(arquivo))
        elif "chatstorage" in nome:
            print(f"[Evidências] Lendo banco iOS: {arquivo.name}")
            mensagens.extend(ler_sqlite_ios(arquivo))

    # 2. Fallback: Exports JSON/TXT
    if not mensagens:
        for arquivo in arquivos:
            if not PADRAO_NOME_CHAT.search(arquivo.name):
                continue

            if arquivo.suffix.lower() == ".json":
                try:
                    dados = json.loads(
                        arquivo.read_text(encoding="utf-8", errors="replace")
                    )
                    if isinstance(dados, list):
                        mensagens.extend(
                            {**mensagem, "arquivo": arquivo.name}
                            for mensagem in dados
                            if isinstance(mensagem, dict)
                        )
                    qtd = len(dados) if isinstance(dados, list) else 0
                    print(f"[Evidências] JSON {arquivo.name}: {qtd} itens")
                except (OSError, json.JSONDecodeError) as erro:
                    print(f"[Evidências] Erro ao ler JSON {arquivo.name}: {erro}")

            elif arquivo.suffix.lower() == ".txt":
                try:
                    texto = arquivo.read_text(encoding="utf-8", errors="replace")
                    encontradas = parse_mensagens(texto, arquivo.name)
                    mensagens.extend(encontradas)
                    print(
                        f"[Evidências] TXT {arquivo.name}: {len(encontradas)} mensagens"
                    )
                except OSError as erro:
                    print(f"[Evidências] Erro ao ler TXT {arquivo.name}: {erro}")

    print(f"[Evidências] Total de mensagens carregadas: {len(mensagens)}")
    return mensagens


def _ler_sqlite(caminho, consulta, parametros=()):
    """Executa consulta SQLite com conexão segura."""
    conexao = None
    try:
        conexao = sqlite3.connect(caminho)
        dados = pd.read_sql_query(consulta, conexao, params=parametros)
        return dados.to_dict(orient="records")
    except (sqlite3.Error, pd.errors.DatabaseError, ValueError) as erro:
        print(f"[Evidências] Erro ao ler SQLite {caminho.name}: {erro}")
        return []
    finally:
        if conexao is not None:
            conexao.close()


def ler_sqlite_android(caminho):
    """Extrai mensagens da tabela messages do msgstore.db (Android)."""
    consulta = """
        SELECT
            datetime(timestamp / 1000, 'unixepoch', 'localtime') AS data_hora,
            CASE
                WHEN key_from_me = 1 THEN 'Investigado'
                WHEN remote_resource IS NOT NULL AND remote_resource != ''
                    THEN remote_resource
                ELSE key_remote_jid
            END AS remetente,
            COALESCE(data, '<Mídia ou conteúdo sem texto>') AS mensagem,
            ? AS arquivo
        FROM messages
        WHERE data IS NOT NULL AND data != '';
    """
    return _ler_sqlite(caminho, consulta, (caminho.name,))


def ler_sqlite_ios(caminho):
    """Extrai mensagens da tabela ZWAMESSAGE do ChatStorage.sqlite (iOS)."""
    consulta = """
        SELECT
            datetime(ZMESSAGEDATE + 978307200, 'unixepoch', 'localtime') AS data_hora,
            COALESCE(ZFROMJID, 'Investigado') AS remetente,
            ZTEXT AS mensagem,
            ? AS arquivo
        FROM ZWAMESSAGE
        WHERE ZTEXT IS NOT NULL AND ZTEXT != '';
    """
    return _ler_sqlite(caminho, consulta, (caminho.name,))


def _achar_coluna(dataframe, candidatos):
    mapa = {str(coluna).strip().lower(): coluna for coluna in dataframe.columns}
    return next((mapa[nome] for nome in candidatos if nome in mapa), None)


def ler_csv(caminho):
    """Lê CSV do IPED (';') ou comum (',') em UTF-8 ou Latin-1."""
    ultimo_erro = None
    for encoding in ("utf-8-sig", "latin-1"):
        for separador in (";", ",", None):
            try:
                dataframe = pd.read_csv(
                    caminho,
                    sep=separador,
                    engine="python",
                    encoding=encoding,
                    dtype=str,
                    keep_default_na=False,
                )
                if dataframe.shape[1] > 1:
                    return dataframe
            except UnicodeDecodeError as erro:
                ultimo_erro = erro
                break
            except (OSError, pd.errors.ParserError, ValueError) as erro:
                ultimo_erro = erro

    raise ValueError(f"Não foi possível ler o CSV {caminho}: {ultimo_erro}")


def _resolver(caminho_relativo, pasta, indice):
    """Resolve caminhos do CSV relativos à raiz do caso ou à pasta de evidências."""
    valor = str(caminho_relativo).strip()
    if not valor:
        return None

    caminho = Path(valor.replace("\\", "/"))
    candidatos = (caminho, pasta / caminho, pasta.parent / caminho, pasta / caminho.name)

    for candidato in candidatos:
        if candidato.is_file():
            return candidato

    return indice.get(caminho.name)


def listar_arquivos(regex, rotulo):
    """Retorna arquivos cuja Categoria, Extensão ou sufixo corresponde ao padrão."""
    pasta = config.EVIDENCE_DIR
    if not pasta.is_dir():
        raise FileNotFoundError(f"Pasta de evidências não encontrada: {pasta}")

    indice = {arquivo.name: arquivo for arquivo in pasta.rglob("*") if arquivo.is_file()}
    encontrados = []
    arquivo_csv = config.FILELIST

    if arquivo_csv.is_file():
        dataframe = ler_csv(arquivo_csv)
        coluna_nome = _achar_coluna(dataframe, COLUNAS_NOME)
        coluna_caminho = _achar_coluna(dataframe, COLUNAS_CAMINHO)
        coluna_categoria = _achar_coluna(dataframe, COLUNAS_CATEGORIA)
        coluna_extensao = _achar_coluna(dataframe, COLUNAS_EXT)
        coluna_arquivo = coluna_caminho or coluna_nome

        if coluna_arquivo is None:
            print(
                f"[{rotulo}] CSV sem coluna de nome/caminho "
                f"({list(dataframe.columns)}). Varrendo a pasta."
            )
        else:
            if coluna_categoria is None and coluna_extensao is None:
                print(f"[{rotulo}] CSV sem Categoria/Ext; usando a extensão do arquivo.")

            for _, linha in dataframe.iterrows():
                caminho = _resolver(linha[coluna_arquivo], pasta, indice)
                if caminho is None or not caminho.is_file():
                    continue

                tipo = " ".join(
                    (
                        str(linha[coluna_categoria]) if coluna_categoria else "",
                        str(linha[coluna_extensao]) if coluna_extensao else "",
                        caminho.suffix,
                    )
                )
                if regex.search(tipo):
                    encontrados.append(caminho)

            print(
                f"[{rotulo}] {arquivo_csv.name}: {len(encontrados)} arquivo(s) encontrados."
            )

    if not encontrados:
        encontrados = [
            arquivo for arquivo in indice.values() if regex.search(arquivo.suffix)
        ]
        print(
            f"[{rotulo}] Varredura de {pasta}: {len(encontrados)} arquivo(s) encontrados."
        )

    return sorted(set(encontrados))


def parse_mensagens(texto, nome_arquivo):
    """Converte linhas TXT tabuladas (IPED) ou nativas do WhatsApp ([data] remetente: mensagem)."""
    mensagens = []

    for linha in str(texto or "").splitlines():
        linha = linha.strip()
        if not linha:
            continue

        # Formato TSV / IPED
        partes = [parte.strip() for parte in linha.split("\t")]
        if len(partes) >= 3:
            if "data" in partes[0].lower() or "date" in partes[0].lower():
                continue

            deslocamento = 1 if partes[0].isdigit() else 0
            if len(partes) > deslocamento + 2:
                mensagens.append(
                    {
                        "data_hora": partes[deslocamento],
                        "remetente": partes[deslocamento + 1],
                        "mensagem": " ".join(partes[deslocamento + 2 :]).strip(),
                        "arquivo": nome_arquivo,
                    }
                )
                continue

        # Formato Nativo WhatsApp TXT
        correspondencia = re.match(
            r"^(?:\[(.*?)\]|(.*?)\s-)\s*([^:]+):\s*(.*)$",
            linha,
        )
        if correspondencia:
            data_hora = correspondencia.group(1) or correspondencia.group(2) or "?"
            mensagens.append(
                {
                    "data_hora": data_hora.strip(),
                    "remetente": (correspondencia.group(3) or "?").strip(),
                    "mensagem": (correspondencia.group(4) or "").strip(),
                    "arquivo": nome_arquivo,
                }
            )

    return mensagens