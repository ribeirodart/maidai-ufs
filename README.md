# maidai-ufs ![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)

Multimodal forensic pipeline / Pipeline forense multimodal

[English](#en) | [Português (Brasil)](#pt-br)

---

<a id="en"></a>

# English

## What it does

The pipeline reads evidence files (images and audio), analyzes each type with a different model, and produces a final forensic report.

1. Evidence discovery: uses the IPED export (a CSV file, parsed with pandas and regex) when present. The file name does not matter (see "Evidence files"). Otherwise it scans the evidence folder.
2. Image stage: Moondream 3.1 analyzes the images and writes `visual.json`.
3. Audio stage: Whisper large v3 turbo transcribes the audio and writes `audio.json`, indicating at which point of the audio (timestamps) each phrase was spoken.
4. Analysis stage: Llama 3.2 served with vLLM combines the results and writes the JSON files and `relatorio_forense_final.txt`.

Each model runs in its own process. The Llama/vLLM stage always runs alone, at the end, because vLLM reserves about 80% of the GPU memory (`gpu_memory_utilization=0.8`).

## Requirements

- Linux, or Windows with WSL, with an NVIDIA GPU. Validated on an NVIDIA RTX 4000 Ada (minimum VRAM not measured). On WSL, the NVIDIA driver is installed on Windows.
- Recent NVIDIA driver (validated with PyTorch 2.13.0 and CUDA 13 libraries). Check with `nvidia-smi`.
- `gpu_memory_utilization` is set to `0.8` (80% of the GPU memory) in `src/llama_vllm.py`. If the GPU is already heavily used, lower this value.
- Python 3.11, pip, venv and git.
- `ffmpeg` in the PATH (required by Whisper).
- Hugging Face account and token, with the Llama 3.2 license accepted on its model page.
- Internet on the first run and tens of GB of free disk space.
- GUI only: graphical display and tkinter (Debian/Ubuntu: `sudo apt install python3-tk`).

The demonstration (`src/demo`) does not support other GPUs (AMD, integrated) or CPU-only yet, since it is only a demonstration of the pipeline flow. The `--demo` mode does not load any model.

## Installation

Run all commands from the repository root (the folder that contains `src/` and `data/`).

### 1. Clone and create a virtual environment

```
git clone <repository-url>
cd maidai-ufs
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Confirm that the environment is active: `which python` must point to `.venv/bin/python`.

### 2. Install the main dependencies

```
pip install -r requirements-local.txt
```

This installs vllm, transformers, accelerate, huggingface_hub, moondream, customtkinter, pandas, numpy and python-dotenv, all with pinned versions. PyTorch is installed automatically as a dependency of vllm. Do not install PyTorch separately.

### 3. Install setuptools and wheel

```
pip install "setuptools<80" wheel
```

### 4. Install Whisper

```
pip install --no-build-isolation openai-whisper==20250625
```

### 5. Install whisper-timestamped

```
pip install whisper-timestamped==1.15.9
```

Steps 3 to 5 are not in `requirements-local.txt` on purpose. `openai-whisper` is built from source and its build needs `pkg_resources`, which recent setuptools versions removed. The `--no-build-isolation` flag only works on the command line, not inside a requirements file. `whisper-timestamped` depends on `openai-whisper`, so it must be installed after it. Keep this order.

### 6. Verify the installation

```
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import vllm, moondream, whisper, whisper_timestamped, customtkinter, pandas; print('imports ok')"
which ffmpeg
```

The first command must print `True` as its second value.

### 7. Log in to Hugging Face

```
hf auth login
```

On older versions of `huggingface_hub`, the command is `huggingface-cli login`. Alternatively, set the `HF_TOKEN` environment variable. Never commit the token or the `.env` file.

## Evidence files

Put the files in `data/evidence/`:

- images and audio files;
- optionally, the IPED export as a CSV file. If there is none, the folder is scanned.

The CSV name does not matter, because the examiner chooses it when exporting from IPED. The pipeline looks for the file in the evidence folder and in the folder above it, in this order:

1. files with a known name (for example `Filelist.csv`);
2. any other `.csv` whose header contains IPED columns such as `Nome`, `Caminho` or `Categoria`.

If nothing matches, it falls back to the default name `Filelist.csv` in the evidence folder.

To use another location:

```
export MAIDAI_EVIDENCE_DIR=/path/to/evidence
export MAIDAI_FILELIST=/path/to/export.csv   # optional, any file name
```

## IPED (optional)

You do not need to run IPED to use this pipeline. This section covers the ready-made demo export and, for those who want to see how an examiner analyzes evidence in IPED, how to get the SearchApp working.

### Ready-made CSV for the demo

The `output_iped/` folder already contains an IPED export with a CSV file ready to be used in the demo. You can test the pipeline without processing any evidence in IPED yourself. The CSV name does not matter: any `.csv` with the IPED header columns is recognized. To use it, either copy it to the evidence folder (or the folder above it), or point to it explicitly:

```
export MAIDAI_FILELIST=/path/to/output_iped/<name-of-the-csv>.csv
```

### SearchApp (IPED output viewer) does not work out of the box

The SearchApp, which is the IPED output viewer, does not work with the files as they are in this repository. To open it and see how the examiner's evidence analysis works in IPED, download:

- [IpedTools](https://www.ipedtools.com.br/downloads)
- [IPED 4.3.1](https://github.com/sepinf-inc/IPED/releases/tag/4.3.1)

Then:

1. Extract IPED 4.3.1.
2. Put IpedTools inside the IPED folder, the one that contains the `iped.exe` and `iped.jar` files.
3. Open IpedTools from inside that folder.
4. Go to Settings > Environment Settings (Configurações > Configurações de Ambiente) > Number of Threads, `N° de Threads (0 = default)`, and increase the value according to your machine.

Attention: with the default value (0) an error occurred. It worked with 4 threads on the author's notebook. If you still get the error, try a different value for your hardware.

## Running

Quick test, no GPU and no models (fictitious data):

```
python src/main.py --demo --no-gui
```

Real pipeline in the terminal (prints the report at the end):

```
python src/main.py --no-gui
```

Before running it, check with `nvidia-smi` that no other process is holding the GPU.

Desktop interface (needs a graphical display):

```
python src/main.py
```

Each stage can also run alone:

```
python src/visual_moondream.py
python src/audio_whisper.py
python src/llama_vllm.py
```

The Llama/vLLM stage uses the results of the previous stages, so run the image and audio stages first.

## Output

All outputs are written to the evidence folder (`data/evidence/` by default):

- `visual.json`: image analysis
- `audio.json`: audio transcription
- other JSON files from the analysis stage
- `relatorio_forense_final.txt`: final report

## Model cache

On the first run, the models are downloaded. Hugging Face models go to `~/.cache/huggingface`. Whisper weights go, by default, to `~/.cache/whisper`. Later runs reuse the cache.

## Troubleshooting

- `No module named 'pkg_resources'` while installing Whisper: run step 3 first and keep `--no-build-isolation` in step 4.
- `torch.cuda.is_available()` returns `False`: there is no NVIDIA GPU, or the driver is missing or outdated. Check `nvidia-smi`.
- vLLM fails to start or runs out of memory: another process is using the GPU. Check `nvidia-smi` and close it, or lower `gpu_memory_utilization` (0.8 by default).
- 401 or 403 when loading Llama 3.2: you are not logged in to Hugging Face, or the model license was not accepted.
- `ffmpeg` not found: install it with your package manager (Debian/Ubuntu: `sudo apt install ffmpeg`).
- Moondream error mentioning `cudaLibraryLoadData` and CUDA 12.8 or newer: the installed PyTorch bundles an older CUDA runtime. Use a fresh virtual environment and the installation steps above. Do not mix other PyTorch installs into the same environment.
- Error when running IpedTools / opening the SearchApp: increase `N° de Threads` in the environment settings (see the IPED section above).

## Known limitations

- The versions in `requirements-local.txt` come from an environment where the full pipeline ran end to end, on one GPU and driver. Other hardware was not tested.
- The IPED SearchApp needs the extra setup described in the IPED section (IpedTools and IPED 4.3.1, with the thread count adjusted).

---

<a id="pt-br"></a>

# Português (Brasil)

## O que faz

O pipeline lê arquivos de evidência (imagens e áudios), analisa cada tipo com um modelo diferente e gera um relatório forense final.

1. Descoberta de arquivos: usa a exportação do IPED (um arquivo CSV, lido com pandas e regex) quando existe. O nome do arquivo não importa (veja "Arquivos de evidência"). Caso contrário, varre a pasta de evidências.
2. Etapa de imagens: o Moondream 3.1 analisa as imagens e grava `visual.json`.
3. Etapa de áudio: o Whisper large v3 turbo transcreve os áudios e grava `audio.json`, indicando em qual momento do áudio (timestamps) cada frase foi falada.
4. Etapa de análise: o Llama 3.2 servido com vLLM combina os resultados e grava os JSONs e o `relatorio_forense_final.txt`.

Cada modelo roda em um processo próprio. A etapa Llama/vLLM sempre roda sozinha, no final, porque o vLLM reserva cerca de 80% da memória da GPU (`gpu_memory_utilization=0.8`).

## Requisitos

- Linux, ou Windows com WSL, com GPU NVIDIA. Validado em uma NVIDIA RTX 4000 Ada (VRAM mínima não medida). No WSL, o driver NVIDIA é instalado no Windows.
- Driver NVIDIA recente (validado com PyTorch 2.13.0 e bibliotecas CUDA 13). Confira com `nvidia-smi`.
- O `gpu_memory_utilization` está em `0.8` (80% da memória da GPU) em `src/llama_vllm.py`. Se a GPU já estiver com muito uso, diminua esse valor.
- Python 3.11, pip, venv e git.
- `ffmpeg` no PATH (exigido pelo Whisper).
- Conta e token do Hugging Face, com a licença do Llama 3.2 aceita na página do modelo.
- Internet na primeira execução e dezenas de GB de disco livre.
- Só para a interface gráfica: tela gráfica e tkinter (Debian/Ubuntu: `sudo apt install python3-tk`).

A demonstração (`src/demo`) ainda não tem suporte para outras GPUs (AMD, integrada) nem para CPU, por ser apenas uma demonstração do fluxo do pipeline. O modo `--demo` não carrega nenhum modelo.

## Instalação

Rode todos os comandos a partir da raiz do repositório (a pasta que contém `src/` e `data/`).

### 1. Clonar e criar o ambiente virtual

```
git clone <url-do-repositorio>
cd maidai-ufs
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Confirme que o ambiente está ativo: `which python` deve apontar para `.venv/bin/python`.

### 2. Instalar as dependências principais

```
pip install -r requirements-local.txt
```

Isso instala vllm, transformers, accelerate, huggingface_hub, moondream, customtkinter, pandas, numpy e python-dotenv, todos com versão fixa. O PyTorch é instalado automaticamente como dependência do vllm. Não instale o PyTorch separadamente.

### 3. Instalar setuptools e wheel

```
pip install "setuptools<80" wheel
```

### 4. Instalar o Whisper

```
pip install --no-build-isolation openai-whisper==20250625
```

### 5. Instalar o whisper-timestamped

```
pip install whisper-timestamped==1.15.9
```

Os passos 3 a 5 ficam fora do `requirements-local.txt` de propósito. O `openai-whisper` é compilado a partir do código-fonte e o build precisa do `pkg_resources`, que as versões recentes do setuptools removeram. A flag `--no-build-isolation` só funciona na linha de comando, não dentro de um arquivo de requirements. O `whisper-timestamped` depende do `openai-whisper`, então precisa ser instalado depois dele. Mantenha essa ordem.

### 6. Verificar a instalação

```
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import vllm, moondream, whisper, whisper_timestamped, customtkinter, pandas; print('imports ok')"
which ffmpeg
```

O primeiro comando deve imprimir `True` como segundo valor.

### 7. Entrar no Hugging Face

```
hf auth login
```

Em versões antigas do `huggingface_hub`, o comando é `huggingface-cli login`. Como alternativa, defina a variável de ambiente `HF_TOKEN`. Nunca versione o token nem o arquivo `.env`.

## Arquivos de evidência

Coloque os arquivos em `data/evidence/`:

- imagens e áudios;
- opcionalmente, a exportação do IPED em CSV. Se não houver nenhum, a pasta é varrida.

O nome do CSV não importa, porque o perito escolhe o nome ao exportar do IPED. O pipeline procura o arquivo na pasta de evidências e na pasta acima dela, nesta ordem:

1. arquivos com nome conhecido (por exemplo `Filelist.csv`);
2. qualquer outro `.csv` cujo cabeçalho tenha colunas do IPED como `Nome`, `Caminho` ou `Categoria`.

Se nada servir, usa o nome padrão `Filelist.csv` na pasta de evidências.

Para usar outro local:

```
export MAIDAI_EVIDENCE_DIR=/caminho/para/evidencias
export MAIDAI_FILELIST=/caminho/para/exportacao.csv   # opcional, qualquer nome
```

## IPED (opcional)

Você não precisa rodar o IPED para usar este pipeline. Esta seção trata da exportação pronta para a demo e, para quem quiser ver como o perito analisa as evidências no IPED, de como fazer o SearchApp funcionar.

### CSV pronto para a demo

A pasta `output_iped/` já contém uma exportação do IPED com um arquivo CSV pronto para ser usado na demo. Assim, você pode testar o pipeline sem precisar processar nenhuma evidência no IPED. O nome do CSV não importa: qualquer `.csv` com as colunas de cabeçalho do IPED é reconhecido. Para usá-lo, copie-o para a pasta de evidências (ou para a pasta acima dela) ou aponte para ele explicitamente:

```
export MAIDAI_FILELIST=/caminho/para/output_iped/<nome-do-csv>.csv
```

### O SearchApp (visualizador da saída do IPED) não funciona de fábrica

O SearchApp, que é o visualizador da saída do IPED, não funciona com os arquivos como estão neste repositório. Para abri-lo e ver como funciona a análise de evidências do perito no IPED, baixe:

- [IpedTools](https://www.ipedtools.com.br/downloads)
- [IPED 4.3.1](https://github.com/sepinf-inc/IPED/releases/tag/4.3.1)

Depois:

1. Extraia o IPED 4.3.1.
2. Coloque o IpedTools dentro da pasta do IPED, aquela que contém os arquivos `iped.exe` e `iped.jar`.
3. Abra o IpedTools de dentro dessa pasta.
4. Vá em Configurações > Configurações de Ambiente > `N° de Threads (0 = default)` e aumente o valor conforme a sua máquina.

Atenção: com o valor padrão (0) ocorreu erro. No notebook do autor funcionou com 4 threads. Se o erro continuar, teste outro valor para o seu hardware.

## Execução

Teste rápido, sem GPU e sem modelos (dados fictícios):

```
python src/main.py --demo --no-gui
```

Pipeline real no terminal (imprime o relatório no final):

```
python src/main.py --no-gui
```

Antes de rodar, confira com `nvidia-smi` que nenhum outro processo está segurando a GPU.

Interface desktop (precisa de tela gráfica):

```
python src/main.py
```

Cada etapa também roda sozinha:

```
python src/visual_moondream.py
python src/audio_whisper.py
python src/llama_vllm.py
```

A etapa Llama/vLLM usa os resultados das etapas anteriores, então rode antes as de imagem e áudio.

## Saída

Tudo é gravado na pasta de evidências (`data/evidence/` por padrão):

- `visual.json`: análise das imagens
- `audio.json`: transcrição dos áudios
- outros JSONs da etapa de análise
- `relatorio_forense_final.txt`: relatório final

## Cache dos modelos

Na primeira execução, os modelos são baixados. Os modelos do Hugging Face vão para `~/.cache/huggingface`. Os pesos do Whisper vão, por padrão, para `~/.cache/whisper`. As execuções seguintes reaproveitam o cache.

## Solução de problemas

- `No module named 'pkg_resources'` ao instalar o Whisper: rode o passo 3 antes e mantenha `--no-build-isolation` no passo 4.
- `torch.cuda.is_available()` retorna `False`: não há GPU NVIDIA, ou o driver está ausente ou desatualizado. Confira `nvidia-smi`.
- O vLLM não inicia ou fica sem memória: outro processo está usando a GPU. Confira `nvidia-smi` e encerre-o, ou diminua o `gpu_memory_utilization` (0.8 por padrão).
- 401 ou 403 ao carregar o Llama 3.2: você não fez login no Hugging Face, ou a licença do modelo não foi aceita.
- `ffmpeg` não encontrado: instale pelo gerenciador de pacotes (Debian/Ubuntu: `sudo apt install ffmpeg`).
- Erro do Moondream citando `cudaLibraryLoadData` e CUDA 12.8 ou superior: o PyTorch instalado traz um runtime CUDA mais antigo. Use um ambiente virtual novo e os passos de instalação acima. Não misture outras instalações de PyTorch no mesmo ambiente.
- Erro ao rodar o IpedTools / abrir o SearchApp: aumente o `N° de Threads` nas configurações de ambiente (veja a seção IPED acima).

## Limitações conhecidas

- As versões do `requirements-local.txt` vêm de um ambiente em que o pipeline completo rodou de ponta a ponta, em uma GPU e um driver. Outros hardwares não foram testados.
- O SearchApp do IPED exige a configuração extra descrita na seção IPED (IpedTools e IPED 4.3.1, com o número de threads ajustado).