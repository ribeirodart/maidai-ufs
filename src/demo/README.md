# maidai-ufs — Pipeline forense multimodal

```
maidai-ufs/
├── requirements.txt
├── data/evidence/                 # imagens, áudios e Filelist.csv; os JSONs e o relatório também saem aqui
└── src/
    ├── config.py                  # caminhos (./maidai-ufs/data/evidence/) e opções
    ├── evidencias.py              # descoberta de arquivos: Filelist.csv (pandas + regex) ou varredura
    ├── visual_moondream.py        # imagens  -> visual.json (foco pericial)
    ├── audio_whisper.py           # áudios   -> audio.json (transcrição pura)
    ├── llama_vllm.py              # análise multimodal -> JSONs + relatorio_forense_final.txt
    ├── relatorio.py               # gera o relatório em Markdown/texto
    ├── pipeline.py                # une as etapas (cada modelo em um processo próprio)
    ├── gui.py                     # interface CustomTkinter
    ├── mock_dados.py              # dados FICTÍCIOS para o modo --demo
    └── main.py                    # ponto de entrada
```

## Instalar
```
pip install -r requirements.txt
pip install --upgrade backports.tarfile
huggingface-cli login
```

## Rodar
```
python maidai-ufs/src/main.py                  # GUI
python maidai-ufs/src/main.py --no-gui         # terminal; imprime o relatório no final
python maidai-ufs/src/main.py --demo           # GUI com dados fictícios (sem modelos)
python maidai-ufs/src/main.py --demo --no-gui  # idem, no terminal
```
Cada módulo (`visual_moondream.py`, `audio_whisper.py`, `llama_vllm.py`) também roda sozinho.

## Docker (para rodar em outro PC)

**Pré-requisitos no PC de destino:** GPU NVIDIA com driver atualizado, Docker e o *NVIDIA Container Toolkit*
(Linux), ou Docker Desktop com WSL2 e GPU habilitada (Windows). Teste: `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`.

```
cp .env.example .env            # coloque seu HF_TOKEN (Llama 3.2 é gated; aceite a licença no Hugging Face)
mkdir -p data/evidence          # coloque aqui imagens, áudios e o CSV exportado do IPED
docker compose build            # demora e baixa vários GB
docker compose run --rm maidai --demo --no-gui   # teste SEM modelos (dados fictícios)
docker compose run --rm maidai                   # pipeline real; relatório impresso e salvo em data/evidence/
```
- Os modelos são baixados na 1ª execução para o volume `maidai-cache` (as seguintes são rápidas). Não ficam na imagem.
- Os JSONs e o `relatorio_forense_final.txt` saem em `./data/evidence` no PC (no Linux podem ficar com dono root).
- **SearchApp.exe não roda no container.** Abra o `IPED-SearchApp.exe` no Windows, na pasta do caso IPED, e use os termos do relatório.
- **Interface gráfica (opcional):** Linux: `xhost +local:docker` e depois `docker compose --profile gui run --rm maidai-gui`.
  No Windows a GUI via container (WSLg) é mais trabalhosa; prefira o terminal (`--no-gui`) ou rodar a GUI direto no Windows.
