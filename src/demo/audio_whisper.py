"""Whisper (whisper-timestamped) — transcrição dos áudios de evidência -> audio.json.

Roda sozinho:  python src/audio_whisper.py

Papel no pipeline: a fala transcrita aqui é lida pelo llama_vllm.py, que a usa para achar entidades,
eventos e termos codificados (gírias como 'parada' ou 'coisa'). Por isso a transcrição precisa ser LITERAL:
nenhum termo de caso entra no prompt inicial (nomes ou palavras suspeitas fariam o Whisper "ouvir" o que
não foi dito) e o texto de cada trecho nunca é reescrito.

Ruído: trecho que parece alucinação do modelo (frase de legenda inventada em silêncio, repetição em loop,
eco do prompt) sai de "timestamps", mas fica em "trechos_descartados" com o motivo, para auditoria.
Trecho com confiança baixa permanece, marcado como "incerto". Falha de leitura vira registro com "erro".

Contrato com o llama_vllm.py (não quebrar):
  - campos lidos: name, full_text, timestamps[].timestamp e timestamps[].text
Campos extras (confianca, incerto, motivos, trechos_descartados, observacao, erro) servem ao perito;
o Llama os ignora.
"""
import gc
import json
import re
import shutil
import sys
import unicodedata
from pathlib import Path

import torch
import whisper_timestamped as whisper

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from evidencias import REGEX_AUDIO, listar_arquivos  # noqa: E402

# Contexto neutro: só o gênero da fala (mensagem de voz informal, português do Brasil) e o tipo de dado
# que costuma aparecer. Ajuda com gírias, grafia de números e nomes próprios. NÃO inclua nomes do caso
# nem termos suspeitos aqui. Para transcrição 100% sem contexto, use USAR_PROMPT_INICIAL = False.
USAR_PROMPT_INICIAL = True
PROMPT_INICIAL = ("Mensagem de voz em português do Brasil, fala informal com gírias, "
                  "nomes próprios, valores em reais e horários.")

# Decodificação: com temperature em tupla, o Whisper refaz o trecho que falha nos limiares abaixo
# (loop de repetição ou baixa probabilidade). condition_on_previous_text=False impede que uma
# alucinação em trecho ruidoso contamine os trechos seguintes.
OPCOES_WHISPER = {
    "language": "pt",
    "beam_size": 5,
    "best_of": 5,
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "compression_ratio_threshold": 2.4,
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "condition_on_previous_text": False,
}
if USAR_PROMPT_INICIAL:
    OPCOES_WHISPER["initial_prompt"] = PROMPT_INICIAL

# Limiares da triagem (confiança = média da confiança das palavras do trecho, de 0 a 1)
LIMIAR_SEM_FALA = 0.6      # no_speech_prob do Whisper
LIMIAR_REPETICAO = 2.4     # compression_ratio do Whisper
LIMIAR_CONFIANCA = 0.5

# Frases que o Whisper inventa em silêncio ou ruído (legendas de vídeo, despedidas de canal, tags)
PADROES_ALUCINACAO = re.compile(
    r"amara\.org|legendas?\s+(pela|por|de)\b|legendado|tradu[cç][aã]o\s+(por|de)\b|inscreva-?se"
    r"|obrigad[oa]\s+por\s+(assistir|ouvir|acompanhar)|^\W*[\[(][^\])]*[\])]\W*$",
    re.I,
)


def _fmt(t):
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


_PROMPT_NORM = _norm(PROMPT_INICIAL)


def _parece_alucinacao(texto):
    n = _norm(texto)
    if PADROES_ALUCINACAO.search(texto):
        return True
    # eco do prompt inicial: o modelo repete o contexto em vez de transcrever
    return USAR_PROMPT_INICIAL and len(n.split()) >= 4 and n in _PROMPT_NORM


def _triagem(texto, conf, sem_fala, compressao):
    """Devolve ('ok'|'incerto'|'descartar', motivos). Descarta só com sinal duplo; um sinal isolado
    mantém o trecho como incerto."""
    motivos = []
    if sem_fala > LIMIAR_SEM_FALA:
        motivos.append("provável silêncio ou ruído sem fala")
    if compressao > LIMIAR_REPETICAO:
        motivos.append("repetição suspeita")
    if conf < LIMIAR_CONFIANCA:
        motivos.append("baixa confiança nas palavras")
    alucinacao = _parece_alucinacao(texto)

    if alucinacao and motivos:
        return "descartar", ["possível alucinação do modelo"] + motivos
    if conf < LIMIAR_CONFIANCA and len(motivos) >= 2:
        return "descartar", motivos
    if alucinacao:
        return "incerto", ["frase típica de alucinação, mas com boa confiança"]
    if motivos:
        return "incerto", motivos
    return "ok", []


def _transcrever(model, audio):
    try:
        return whisper.transcribe(model, audio, **OPCOES_WHISPER)
    except TypeError as e:
        print(f"[Whisper] Aviso: opção de decodificação não aceita ({e}); usando opções mínimas.")
        return whisper.transcribe(model, audio, language="pt")


def _processar_segmentos(res):
    segmentos, descartados = [], []
    for seg in res.get("segments", []):
        start, end = seg.get("start", 0), seg.get("end", 0)
        texto = seg.get("text", "").strip()
        if not texto:
            continue
        ts = f"[{_fmt(start)} -> {_fmt(end)}]"
        palavras = [
            {
                "word": w.get("text", w.get("word")),
                "start": round(w.get("start", 0), 2),
                "end": round(w.get("end", 0), 2),
                "confidence": round(w.get("confidence") or 0, 2),
            }
            for w in seg.get("words", [])
        ]
        confs = [p["confidence"] for p in palavras]
        conf = sum(confs) / len(confs) if confs else float(seg.get("confidence") or 0)

        decisao, motivos = _triagem(texto, conf, seg.get("no_speech_prob") or 0,
                                    seg.get("compression_ratio") or 0)
        if decisao == "descartar":
            print(f"[Whisper] {ts} descartado ({'; '.join(motivos)}): {texto}")
            descartados.append({"timestamp": ts, "text": texto, "confianca": round(conf, 2),
                                "motivos": motivos})
            continue

        item = {"timestamp": ts, "start": round(start, 2), "end": round(end, 2), "text": texto,
                "confianca": round(conf, 2), "words": palavras}
        if decisao == "incerto":
            item["incerto"] = True
            item["motivos"] = motivos
        print(f"[Whisper] {ts}{' (incerto)' if decisao == 'incerto' else ''} {texto}")
        segmentos.append(item)
    return segmentos, descartados


def processar_audios_whisper():
    audios = listar_arquivos(REGEX_AUDIO, "Whisper")
    if not audios:
        print("[Whisper] Nenhum áudio para processar.")
        return []

    if shutil.which("ffmpeg") is None:
        print("[Whisper] AVISO: ffmpeg não encontrado no PATH; .ogg/.mp3 podem falhar.")

    print("[Whisper] Carregando modelo na VRAM...")
    model = whisper.load_model("large-v3-turbo", device="cuda")

    resultado = []
    for caminho in audios:
        print(f"[Whisper] Transcrevendo: {caminho.name}")
        try:
            audio = whisper.load_audio(str(caminho))
            torch.manual_seed(0)  # o fallback de temperatura sorteia; semente fixa mantém o resultado repetível
            res = _transcrever(model, audio)
            segmentos, descartados = _processar_segmentos(res)

            item = {
                "name": caminho.name,
                "path": str(caminho),
                # montado só com os trechos mantidos: o res["text"] do Whisper incluiria as alucinações
                "full_text": " ".join(s["text"] for s in segmentos),
                "timestamps": segmentos,
                "confianca_media": round(sum(s["confianca"] for s in segmentos) / len(segmentos), 2) if segmentos else 0.0,
                "trechos_descartados": descartados,
            }
            if not segmentos:
                item["observacao"] = ("Nenhuma fala identificável (silêncio, ruído ou música). "
                                      "Conferir os trechos descartados e ouvir o áudio.")
            resultado.append(item)
            print(f"[Whisper] {caminho.name}: {len(segmentos)} trecho(s), "
                  f"{sum(1 for s in segmentos if s.get('incerto'))} incerto(s), {len(descartados)} descartado(s)")
        except Exception as e:
            # o áudio continua no registro: o perito precisa saber que ele não foi transcrito
            print(f"[Whisper] Erro em {caminho.name}: {e}")
            resultado.append({"name": caminho.name, "path": str(caminho), "full_text": "", "timestamps": [],
                              "trechos_descartados": [], "erro": f"{type(e).__name__}: {e}"})

    saida = config.OUTPUT_DIR / "audio.json"
    with open(saida, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=4)
    print(f"[Whisper] {len(resultado)} item(ns) salvos em {saida}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("[Whisper] Modelo descarregado da VRAM.")
    return resultado


if __name__ == "__main__":
    processar_audios_whisper()