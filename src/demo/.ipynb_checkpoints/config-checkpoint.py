"""Caminhos e opções da aplicação (execução local).

A pasta de evidências padrão é ./maidai-ufs/data/evidence/ (relativa ao diretório
de onde você roda o programa). O CSV exportado pelo IPED (Filelist) agora é buscado
diretamente na pasta 'output_iped/' localizada na raiz do projeto.

Os valores podem ser trocados por variáveis de ambiente (a GUI usa isso para passar
a escolha do usuário aos módulos, que rodam em processos separados):
    MAIDAI_EVIDENCE_DIR  -> pasta das evidências
    MAIDAI_OUTPUT_IPED   -> pasta onde fica o Filelist exportado pelo IPED
    MAIDAI_FILELIST      -> caminho exato do Filelist.csv
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

# Nova pasta padrão para o IPED exportado: maidai-ufs/output_iped/ (ou PROJECT_DIR/output_iped)
def output_iped_dir_padrao() -> Path:
    relativa = Path("./maidai-ufs/output_iped").resolve()
    if relativa.is_dir():
        return relativa
    return (PROJECT_DIR / "output_iped").resolve()

OUTPUT_IPED_DIR = Path(os.environ.get("MAIDAI_OUTPUT_IPED") or output_iped_dir_padrao()).expanduser().resolve()

NOMES_CSV_CONHECIDOS = ["Filelist.csv", "Lista_de_Arquivos.csv", "Lista de Arquivos.csv", "FileList.csv"]
CABECALHO_IPED = ("nome", "name", "caminho", "path", "categoria", "category")


def achar_filelist(pasta_output_iped, pasta_evidencias) -> Path:
    """Acha o CSV exportado pelo IPED. Procura primariamente na pasta 'output_iped'
    (e na pasta de evidências como fallback caso o perito tenha salvo lá):
    1) nomes conhecidos; 2) qualquer .csv cujo cabeçalho tenha Nome/Caminho/Categoria.
    Se nada servir, devolve o caminho padrão dentro de output_iped."""
    
    pastas_busca = [Path(pasta_output_iped), Path(pasta_evidencias), Path(pasta_output_iped).parent]
    
    # 1) Procura por nomes conhecidos nas pastas de busca
    for base in pastas_busca:
        if not base.is_dir():
            continue
        for nome in NOMES_CSV_CONHECIDOS:
            if (base / nome).is_file():
                return base / nome

    # 2) Procura por arquivos .csv com cabeçalho compatível com o IPED
    for base in pastas_busca:
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
                
    # Fallback final apontando para output_iped/Filelist.csv
    return Path(pasta_output_iped) / "Filelist.csv"


FILELIST = Path(os.environ.get("MAIDAI_FILELIST") or achar_filelist(OUTPUT_IPED_DIR, EVIDENCE_DIR)).expanduser()

# Todos los JSONs (visual.json, audio.json, ...) continuam sendo gravados na pasta de evidências
OUTPUT_DIR = EVIDENCE_DIR

# Moondream e Whisper ao mesmo tempo (cada um em seu processo). Só ative se a VRAM
# comportar os dois juntos; o Llama/vLLM sempre roda sozinho depois, pois reserva 92% da GPU.
PARALELO_MOONDREAM_WHISPER = False