"""Moondream — análise visual de foco pericial das imagens de evidência -> visual.json.

Roda sozinho:  python src/visual_moondream.py

Papel no pipeline: o texto gerado aqui é lido pelo llama_vllm.py, que extrai entidades, eventos,
termos codificados e relações. Por isso o Moondream precisa (1) transcrever fielmente nomes, valores,
horários, coordenadas, identificadores e carimbos de documento e (2) apontar itens de interesse SEM afirmá-los.

Ruído: nenhuma imagem é descartada. Cada uma recebe o tipo (foto, captura de tela, documento), uma frase
neutra de contexto e a transcrição do texto; ruído de interface, brinquedos ou desenhos são filtrados nas
perguntas, não na saída. Falha de leitura vira um registro com "erro".

Contexto da investigação: o Moondream não ouve o áudio (as imagens são processadas antes). Para ele olhar a
imagem com o que foi dito em mente, escreva a hipótese do perito em <pasta de evidências>/contexto_filtro.txt
(o mesmo arquivo que o llama_vllm.py já lê). Ela entra como pergunta fechada e a resposta é marcada como
não verificada. Sem o arquivo, nada muda.

Coordenadas: lidas por código do EXIF da imagem (não dependem do modelo) e, quando estampadas na própria
foto, pelo prompt de texto. O local NÃO é interpretado: só indicado para conferência em mapa.

Contrato com o llama_vllm.py (não quebrar):
  - campos lidos: name, description, visible_text
  - SEM_INDICIO tem que ser exatamente o texto abaixo: o llama_vllm.py o cita nas regras do prompt
  - description que NÃO começa com SEM_INDICIO é tratada como conteúdo a analisar
Campos extras (tipo_imagem, indicios, gps, revisar, erro) servem ao perito; o Llama os ignora.
"""
import gc
import json
import re
import sys
from pathlib import Path

import torch
from PIL import Image
import moondream as md

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from evidencias import REGEX_IMAGEM, listar_arquivos  # noqa: E402

SEM_INDICIO = "No suspicious elements visible."   # NÃO ALTERAR (ver contrato acima)
PREFIXO_ITENS = "Items of interest reported by the model (unverified, check the image):"
MAX_CONTEXTO = 1000

# ---------------------------------------------------------------------------
# Prompts. Moondream responde melhor a perguntas curtas e fechadas; por isso cada tarefa é uma
# pergunta, e o código monta a descrição final.
# ---------------------------------------------------------------------------
PROMPT_TIPO = (
    "Classify this image. Answer with exactly ONE word from this list: photo, screenshot, document, other. "
    "photo = a real-world scene taken by a camera. "
    "screenshot = a capture of a phone or computer screen (chat, bank app, social network, map, website). "
    "document = a scanned or photographed paper, ID, receipt or form. "
    "other = anything else (meme, drawing, blank, very dark or blurred image)."
)

PROMPT_CENA = (
    "You are assisting a forensic examiner. In at most three short neutral sentences, describe only what is "
    "clearly visible: the setting (indoor or outdoor, type of place), how many people are visible and what "
    "they are doing, and every object that could matter to an investigation (boxes, packages, bags, "
    "vehicles, phones, documents, money). Mention each such object even if it seems ordinary. "
    "Do not guess identities, intentions or events. Do not describe clothing colors, decoration or "
    "furniture. If the image is too dark, blurred or cropped to tell something, say what cannot be told "
    "instead of guessing."
)

PROMPT_TELA = (
    "In ONE sentence, say what this screenshot or document is (for example: chat conversation, payment "
    "receipt, bank statement, social media profile, map, identity document, handwritten note) and name the "
    "app, institution, people or user handles shown, if any. If it is marked as a sample, simulation, void "
    "or fictional, say so. Do not describe colors or layout."
)

PROMPT_TEXTO = (
    "Transcribe ALL legible text in this image exactly as written, keeping the original language and line "
    "breaks. This includes chat messages, payment receipts, bank apps, documents, handwriting, license "
    "plates, street signs, labels on packages, and camera overlays stamped on a photo (date, time, GPS "
    "coordinates, address, place name). "
    "For chat screenshots, put each message on its own line, starting with who wrote it (the name if shown, "
    "otherwise 'left bubble' or 'right bubble') and the time if visible. "
    "Keep names, nicknames, usernames, phone numbers, amounts, dates, times, coordinates, addresses and "
    "transaction IDs exactly as shown. "
    "Transcribe stamps, banners, footers and watermarks too (for example 'sample', 'simulation', 'void', "
    "'not valid'): they change the meaning of a document. "
    "Ignore only interface noise: status bar, battery, signal, navigation buttons and keyboard keys. "
    "If part of a word or message is illegible, write [illegible] in its place and keep the rest. "
    "Never complete, translate or invent text. "
    "If there is no legible text, answer exactly: none."
)

_CRITERIO = (
    " Answer 'no' unless it is clearly and unambiguously visible. Toys, drawings, cartoons, video games, "
    "posters and objects shown on a screen do not count. If yes, name what you see and where in the image "
    "in one short sentence. If not, answer exactly: no."
)

PERGUNTAS_ITENS = {
    "weapon": "Is a real weapon visible (gun, firearm, ammunition, or a blade being held or displayed; "
              "kitchen utensils in normal use do not count)?" + _CRITERIO,
    "drugs": "Are drugs, drug packaging (small bags, wrapped bundles, pills, powder) or drug "
             "paraphernalia visible?" + _CRITERIO,
    "cash": "Are bundles or large amounts of cash visible?" + _CRITERIO,
    "package": "Is a package, parcel or cardboard box visible (delivery box, taped or wrapped box, bag of "
               "goods)? Also say whether it looks sealed or open and read any label or writing on it."
               + _CRITERIO,
}

_PROMPT_CONTEXTO = (
    "Do not assume this is true; it is only a lead supplied by the forensic examiner: {contexto}\n"
    "Does this image clearly show anything that relates to that lead (a place, an object, a person, a "
    "document, coordinates)? If yes, describe exactly what is visible and how it relates, in one or two "
    "sentences, without adding anything that is not in the image. If not, answer exactly: no."
)

_TIPOS = {"photo": "photo", "photograph": "photo", "picture": "photo",
          "screenshot": "screenshot", "screen": "screenshot",
          "document": "document", "receipt": "document", "paper": "document",
          "other": "other"}
_ROTULO_TIPO = {"screenshot": "Screenshot", "document": "Document"}


# ---------------------------------------------------------------------------
# Auxiliares
# ---------------------------------------------------------------------------
def _perguntar(model, image, pergunta):
    res = model.query(image, pergunta)
    resposta = res.get("answer", "") if isinstance(res, dict) else str(res)
    return resposta.strip()


def _tipo_imagem(resposta):
    """Primeira palavra da resposta; na dúvida, 'photo' (assim a checagem de itens roda, não é pulada)."""
    palavras = re.split(r"\W+", resposta.strip().lower())
    return _TIPOS.get(palavras[0] if palavras else "", "photo")


def _negativo(resposta):
    """True se a resposta nega o item ('no', 'none', 'There is no firearm...') ou vem vazia."""
    t = resposta.strip().lower()
    if not t:
        return True
    return bool(re.search(r"\b(no|not|none|nothing|n/a)\b", " ".join(t.split()[:4])))


def _sem_texto(texto):
    return texto.strip().lower().strip(" .") in ("", "none", "no text", "n/a")


def _contexto_examinador():
    p = config.OUTPUT_DIR / "contexto_filtro.txt"
    try:
        return p.read_text(encoding="utf-8").strip()[:MAX_CONTEXTO] if p.is_file() else ""
    except OSError:
        return ""


def _gps_exif(img):
    """Latitude/longitude do EXIF em graus decimais, ou None. Lido por código, sem modelo."""
    try:
        gps = img.getexif().get_ifd(0x8825)  # GPSInfo
        if not gps:
            return None

        def graus(valor, ref):
            d, m, s = (float(x) for x in valor)
            r = d + m / 60 + s / 3600
            return -r if str(ref).upper() in ("S", "W") else r

        return {"latitude": round(graus(gps[2], gps[1]), 6), "longitude": round(graus(gps[4], gps[3]), 6)}
    except (KeyError, TypeError, ValueError, ZeroDivisionError, AttributeError):
        return None


def _analisar(model, image, contexto, gps):
    tipo = _tipo_imagem(_perguntar(model, image, PROMPT_TIPO))
    texto = _perguntar(model, image, PROMPT_TEXTO)
    visible_text = "" if _sem_texto(texto) else texto

    itens = {}
    if tipo in ("screenshot", "document"):
        # captura de tela ou documento: não faz sentido contar pessoas nem procurar armas
        base = f"{_ROTULO_TIPO[tipo]}: {_perguntar(model, image, PROMPT_TELA)}"
        foto = False
    else:
        base = _perguntar(model, image, PROMPT_CENA)
        foto = True
        for chave, pergunta in PERGUNTAS_ITENS.items():
            resposta = _perguntar(model, image, pergunta)
            if not _negativo(resposta):
                itens[chave] = resposta
    if contexto:
        resposta = _perguntar(model, image, _PROMPT_CONTEXTO.replace("{contexto}", contexto))
        if not _negativo(resposta):
            itens["context"] = resposta

    partes = ""
    if itens:
        partes = PREFIXO_ITENS + " " + "; ".join(f"{k}: {v.strip().rstrip('.')}" for k, v in itens.items()) + "."
    gps_txt = (f"GPS metadata (EXIF): latitude {gps['latitude']}, longitude {gps['longitude']}. "
               "Location not interpreted; verify on a map.") if gps else ""

    if foto and not partes and not gps_txt:
        descricao = f"{SEM_INDICIO} {base}"
    else:
        descricao = " ".join(x for x in (partes, base, gps_txt) if x)

    return {"description": descricao, "visible_text": visible_text, "tipo_imagem": tipo,
            "indicios": itens, "gps": gps, "revisar": bool(itens)}


# ---------------------------------------------------------------------------
# Processamento
# ---------------------------------------------------------------------------
def processar_imagens_moondream():
    imagens = listar_arquivos(REGEX_IMAGEM, "Moondream")
    if not imagens:
        print("[Moondream] Nenhuma imagem para processar.")
        return []

    contexto = _contexto_examinador()
    if contexto:
        print("[Moondream] Contexto do perito carregado (contexto_filtro.txt); respostas ligadas a ele ficam marcadas como não verificadas.")

    print("[Moondream] Carregando modelo na VRAM...")
    model = md.vl(local=True, model="moondream3.1-9B-A2B")

    resultado = []
    for caminho in imagens:
        print(f"[Moondream] Analisando: {caminho.name}")
        try:
            with Image.open(caminho) as bruta:
                gps = _gps_exif(bruta)          # o convert abaixo descarta o EXIF
                image = bruta.convert("RGB")
            analise = _analisar(model, image, contexto, gps)
            resultado.append({"name": caminho.name, "path": str(caminho), **analise})
            if analise["indicios"]:
                print(f"[Moondream] {caminho.name}: itens para revisão manual: {', '.join(analise['indicios'])}")
            if gps:
                print(f"[Moondream] {caminho.name}: GPS no EXIF: {gps['latitude']}, {gps['longitude']}")
        except Exception as e:
            # a imagem continua no registro: o perito precisa saber que ela não foi analisada
            print(f"[Moondream] Erro em {caminho.name}: {e}")
            resultado.append({
                "name": caminho.name, "path": str(caminho),
                "description": f"Image could not be analyzed ({type(e).__name__}); review manually.",
                "visible_text": "", "tipo_imagem": "desconhecido", "indicios": {}, "gps": None,
                "revisar": True, "erro": f"{type(e).__name__}: {e}",
            })

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