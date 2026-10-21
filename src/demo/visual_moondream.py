"""Moondream — análise visual de foco pericial das imagens de evidência -> visual.json.

Roda sozinho:  python src/visual_moondream.py
"""
import gc
import json
import sys
from pathlib import Path

import torch
from PIL import Image
import moondream as md

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from evidencias import REGEX_IMAGEM, listar_arquivos  # noqa: E402

SEM_INDICIO = "No suspicious elements visible."

# Foco pericial estrito + antialucinação
PROMPT_CENA = (
    "You are assisting a forensic examiner. Report ONLY what is clearly visible, and ONLY these topics: "
    "(1) how many people are visible and what each is doing; "
    "(2) visible weapons or firearms; "
    "(3) visible drugs, drug packaging or paraphernalia; "
    "(4) visible bundles of cash or money. "
    "Do NOT describe clothing colors, decoration, furniture, background aesthetics or any other irrelevant detail. "
    "Do NOT guess, infer or invent anything that is not clearly visible. "
    f"If none of topics 2-4 is clearly visible, start your answer with '{SEM_INDICIO}' and then give ONE short, "
    "neutral sentence about the setting and the number of people."
)

PROMPT_TEXTO = (
    "Transcribe exactly any legible text shown on screens (chat conversations, PIX or payment receipts, "
    "bank apps) or on documents in this image. Copy only what you can actually read; do not complete or guess "
    "missing words. If there is no legible text, answer exactly: none."
)


def _perguntar(model, image, pergunta):
    res = model.query(image, pergunta)
    resposta = res.get("answer", "") if isinstance(res, dict) else str(res)
    return resposta.strip()


def processar_imagens_moondream():
    imagens = listar_arquivos(REGEX_IMAGEM, "Moondream")
    if not imagens:
        print("[Moondream] Nenhuma imagem para processar.")
        return []

    print("[Moondream] Carregando modelo na VRAM...")
    model = md.vl(local=True, model="moondream3.1-9B-A2B")

    resultado = []
    for caminho in imagens:
        print(f"[Moondream] Analisando: {caminho.name}")
        try:
            image = Image.open(caminho).convert("RGB")
            cena = _perguntar(model, image, PROMPT_CENA)
            texto = _perguntar(model, image, PROMPT_TEXTO)
            resultado.append({
                "name": caminho.name,
                "path": str(caminho),
                "description": cena,
                "visible_text": "" if texto.strip().lower().strip(".") == "none" else texto,
            })
        except Exception as e:
            print(f"[Moondream] Erro em {caminho.name}: {e}")

    saida = config.OUTPUT_DIR / "visual.json"
    with open(saida, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=4)
    print(f"[Moondream] {len(resultado)} item(ns) salvos em {saida}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("[Moondream] Modelo descarregado da VRAM.")
    return resultado


if __name__ == "__main__":
    processar_imagens_moondream()
