"""Lógica de união da pipeline.

Cada etapa de modelo roda em um PROCESSO separado (python -u src/<modulo>.py). Assim a VRAM é
totalmente liberada ao fim de cada etapa (importante para o vLLM, que reserva 92% da GPU) e
cada módulo continua 100% independente.
"""
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

_PROCS = []
_LOCK = threading.Lock()

# Arquivos gerados pela pipeline (apagados no início de cada execução para não mostrar resultado antigo)
SAIDAS = ["visual.json", "audio.json", "entities.json", "linguagem_codificada.json", "events.json",
          "entity_event_relationships.json", "consolidado.json", "forensic_summary.json",
          "relatorio_forense_final.txt"]


def modo_demo():
    return os.environ.get("MAIDAI_DEMO") == "1"


def pasta_saida(evidence_dir):
    """Em modo demo, os resultados ficam na pasta temporária de demonstração."""
    if modo_demo() and os.environ.get("MAIDAI_DEMO_DIR"):
        return Path(os.environ["MAIDAI_DEMO_DIR"])
    return Path(evidence_dir)


def encerrar_processos():
    with _LOCK:
        for p in _PROCS:
            if p.poll() is None:
                p.terminate()


def _rodar_etapa(titulo, script, env, log):
    log(f"=== {titulo} ===")
    proc = subprocess.Popen(
        [sys.executable, "-u", str(config.SRC_DIR / script)],
        cwd=str(config.SRC_DIR), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    with _LOCK:
        _PROCS.append(proc)
    for linha in proc.stdout:
        log(linha.rstrip("\n"))
    codigo = proc.wait()
    if codigo != 0:
        log(f"[ERRO] {titulo} terminou com código {codigo}.")
    return codigo == 0


def executar_pipeline(evidence_dir, filelist=None, log=print):
    """Roda Moondream -> Whisper -> Llama. Retorna True se o relatório final foi gerado."""
    if modo_demo():
        import mock_dados
        return mock_dados.executar_demo(pasta_saida(evidence_dir), log)

    evidence_dir = Path(evidence_dir).expanduser().resolve()
    filelist = Path(filelist) if filelist else config.achar_filelist(evidence_dir)
    if not evidence_dir.is_dir():
        log(f"[ERRO] Pasta de evidências não encontrada: {evidence_dir}")
        return False

    for nome in SAIDAS:
        try:
            (evidence_dir / nome).unlink(missing_ok=True)
        except OSError:
            pass

    env = dict(os.environ)
    env.update({
        "MAIDAI_EVIDENCE_DIR": str(evidence_dir),
        "MAIDAI_FILELIST": str(filelist),
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
    })
    log(f"[Pipeline] Evidências: {evidence_dir}")
    log(f"[Pipeline] Filelist: {filelist} ({'ok' if filelist.is_file() else 'não encontrado — varredura da pasta'})")

    etapas_visuais = [("ETAPA 1: IMAGENS (MOONDREAM)", "visual_moondream.py"),
                      ("ETAPA 2: ÁUDIOS (WHISPER)", "audio_whisper.py")]
    if config.PARALELO_MOONDREAM_WHISPER:
        threads = [threading.Thread(target=_rodar_etapa, args=(t, s, env, log)) for t, s in etapas_visuais]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    else:
        for t, s in etapas_visuais:
            _rodar_etapa(t, s, env, log)

    if not (ler_json(evidence_dir, "audio.json") or ler_json(evidence_dir, "visual.json")):
        log("[ERRO] Nenhuma saída de áudio/imagem; análise do Llama cancelada.")
        return False

    _rodar_etapa("ETAPA 3: ANÁLISE FORENSE (LLAMA 3.2 / vLLM)", "llama_vllm.py", env, log)
    ok = (evidence_dir / "relatorio_forense_final.txt").is_file()
    log("=== PIPELINE CONCLUÍDA ===" if ok else "=== PIPELINE TERMINOU SEM RELATÓRIO ===")
    return ok


# ------------------------------------------------------------------
# Resultados
# ------------------------------------------------------------------
def ler_json(pasta, nome):
    p = Path(pasta) / nome
    if not p.is_file():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def formatar_relatorio(pasta):
    """Conteúdo de relatorio_forense_final.txt (Markdown/texto)."""
    p = Path(pasta) / "relatorio_forense_final.txt"
    if not p.is_file():
        return "Relatório não gerado."
    return p.read_text(encoding="utf-8")


def texto_imagens(pasta):
    partes = []
    for i in (ler_json(pasta, "visual.json") or []):
        txt = f"{i.get('name')}\n{i.get('description')}"
        if i.get("visible_text"):
            txt += f"\nTexto visível: {i['visible_text']}"
        partes.append(txt)
    return "\n\n".join(partes)


def texto_audios(pasta):
    return "\n\n".join(
        f"{a.get('name')}\n" + "\n".join(f"{s['timestamp']} {s['text']}" for s in a.get("timestamps", []))
        for a in (ler_json(pasta, "audio.json") or [])
    )


# ------------------------------------------------------------------
# SearchApp
# ------------------------------------------------------------------
def abrir_searchapp(caminho):
    """Retorna (ok, mensagem)."""
    if not caminho or not os.path.isfile(caminho):
        return False, "SearchApp.exe não informado ou não encontrado."
    cmd = [caminho]
    if not sys.platform.startswith("win"):
        wine = shutil.which("wine")
        if not wine:
            return False, "Não é Windows e o Wine não foi encontrado; não dá para abrir o .exe."
        cmd = [wine, caminho]
    try:
        subprocess.Popen(cmd, cwd=os.path.dirname(caminho))
        return True, "SearchApp aberto."
    except Exception as e:
        return False, f"Falha ao abrir o SearchApp: {e}"
