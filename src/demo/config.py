"""Caminhos e opções da aplicação (execução local).

A pasta de evidências padrão é ./maidai-ufs/data/evidence/ (relativa ao diretório
de onde você roda o programa). Se ela não existir, usa <repositório>/data/evidence/,
que é a mesma pasta quando o código está em maidai-ufs/src/.

Os valores podem ser trocados por variáveis de ambiente (a GUI usa isso para passar
a escolha do usuário aos módulos, que rodam em processos separados):
    MAIDAI_EVIDENCE_DIR  -> pasta das evidências
    MAIDAI_FILELIST      -> caminho do Filelist.csv
"""
import os
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parents[1]


def evidence_dir_padrao() -> Path:
    relativa = Path("./maidai-ufs/data/evidence").resolve()
    if relativa.is_dir():
        return relativa
    return (PROJECT_DIR / "data" / "evidence").resolve()


EVIDENCE_DIR = Path(os.environ.get("MAIDAI_EVIDENCE_DIR") or evidence_dir_padrao()).expanduser().resolve()
NOMES_CSV_CONHECIDOS = ["Filelist.csv", "Lista_de_Arquivos.csv", "Lista de Arquivos.csv", "FileList.csv"]
CABECALHO_IPED = ("nome", "name", "caminho", "path", "categoria", "category")


def achar_filelist(pasta) -> Path:
    """Acha o CSV exportado pelo IPED. O perito escolhe o nome, então:
    1) nomes conhecidos; 2) qualquer .csv cujo cabeçalho tenha Nome/Caminho/Categoria
    (na pasta de evidências e na pasta acima dela). Se nada servir, devolve o nome padrão."""
    pasta = Path(pasta)
    for base in (pasta, pasta.parent):
        for nome in NOMES_CSV_CONHECIDOS:
            if (base / nome).is_file():
                return base / nome
    for base in (pasta, pasta.parent):
        if not base.is_dir():
            continue
        for csv in sorted(base.glob("*.csv")):
            try:
                cab = csv.open(encoding="utf-8-sig", errors="replace").readline().lower()
            except OSError:
                continue
            if any(f'"{c}"' in cab or cab.startswith(c) or f";{c}" in cab or f",{c}" in cab
                   for c in CABECALHO_IPED):
                return csv
    return pasta / "Filelist.csv"


FILELIST = Path(os.environ.get("MAIDAI_FILELIST") or achar_filelist(EVIDENCE_DIR)).expanduser()

# Todos os JSONs (visual.json, audio.json, ...) são gravados na pasta de evidências
OUTPUT_DIR = EVIDENCE_DIR

# Moondream e Whisper ao mesmo tempo (cada um em seu processo). Só ative se a VRAM
# comportar os dois juntos; o Llama/vLLM sempre roda sozinho depois, pois reserva 92% da GPU.
PARALELO_MOONDREAM_WHISPER = False
