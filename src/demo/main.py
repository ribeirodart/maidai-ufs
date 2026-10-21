"""Ponto de entrada.

    python maidai-ufs/src/main.py                 # abre a GUI
    python maidai-ufs/src/main.py --no-gui        # terminal (imprime o relatório ao final)
    python maidai-ufs/src/main.py --demo          # GUI com dados FICTÍCIOS, sem carregar modelos
    python maidai-ufs/src/main.py --demo --no-gui # idem, no terminal
    python maidai-ufs/src/main.py --evidence PASTA --csv Filelist.csv --searchapp SearchApp.exe
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path


def checar_ambiente():
    try:
        import backports.tarfile  # noqa: F401
    except ImportError:
        print("[Aviso] backports.tarfile ausente. Corrija com: pip install --upgrade backports.tarfile")


def main():
    ap = argparse.ArgumentParser(description="Pipeline forense multimodal")
    ap.add_argument("--no-gui", action="store_true", help="roda sem interface gráfica")
    ap.add_argument("--demo", action="store_true", help="dados fictícios, sem carregar modelos")
    ap.add_argument("--evidence", default=None, help="pasta de evidências")
    ap.add_argument("--csv", default=None, help="caminho do Filelist.csv")
    ap.add_argument("--searchapp", default=None, help="caminho do SearchApp.exe")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    demo_dir = None
    if args.demo:
        demo_dir = tempfile.mkdtemp(prefix="maidai_demo_")
        os.environ["MAIDAI_DEMO"] = "1"
        os.environ["MAIDAI_DEMO_DIR"] = demo_dir
        if not args.evidence:
            args.evidence = demo_dir
        import mock_dados
        mock_dados.criar_evidencias_demo(demo_dir)
        print(f"[DEMO] Pasta temporária com dados fictícios: {demo_dir}")

    # precisa vir antes de importar config
    if args.evidence:
        os.environ["MAIDAI_EVIDENCE_DIR"] = args.evidence
    if args.csv:
        os.environ["MAIDAI_FILELIST"] = args.csv

    import config
    import pipeline

    if not args.demo:
        checar_ambiente()

    if not args.no_gui:
        try:
            from gui import App
            App(config.EVIDENCE_DIR, args.csv, args.searchapp).mainloop()
            return
        except Exception as e:
            print(f"[GUI] Indisponível ({e}). Seguindo no terminal.")

    pipeline.executar_pipeline(config.EVIDENCE_DIR, config.FILELIST)
    print("\n" + "=" * 60)
    print(pipeline.formatar_relatorio(pipeline.pasta_saida(config.EVIDENCE_DIR)))
    if args.searchapp:
        print(pipeline.abrir_searchapp(args.searchapp)[1])


if __name__ == "__main__":
    main()
