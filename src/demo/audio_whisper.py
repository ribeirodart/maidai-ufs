"""Whisper (whisper-timestamped) — transcrição dos áudios de evidência -> audio.json.

Roda sozinho:  python src/audio_whisper.py
"""
import gc
import json
import shutil
import sys
from pathlib import Path

import torch
import whisper_timestamped as whisper

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from evidencias import REGEX_AUDIO, listar_arquivos  # noqa: E402


def _fmt(t):
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


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
            # Transcrição pura: NENHUM prompt/initial_prompt é passado ao Whisper.
            res = whisper.transcribe(model, audio, language="pt")

            segmentos = []
            for seg in res.get("segments", []):
                start, end = seg.get("start", 0), seg.get("end", 0)
                texto = seg.get("text", "").strip()
                ts = f"[{_fmt(start)} -> {_fmt(end)}]"
                print(f"[Whisper] {caminho.name} {ts} {texto}")
                palavras = [
                    {
                        "word": w.get("text", w.get("word")),
                        "start": round(w.get("start", 0), 2),
                        "end": round(w.get("end", 0), 2),
                        "confidence": round(w.get("confidence", 0), 2),
                    }
                    for w in seg.get("words", [])
                ]
                segmentos.append({
                    "timestamp": ts,
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "text": texto,
                    "words": palavras,
                })

            resultado.append({
                "name": caminho.name,
                "path": str(caminho),
                "full_text": res.get("text", "").strip(),
                "timestamps": segmentos,
            })
        except Exception as e:
            print(f"[Whisper] Erro em {caminho.name}: {e}")

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
