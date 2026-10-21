# **MAIDAI Argos Project**  

### Pipeline with AI perception and reasoning models
The porpouse is analysis and make a report with the suspect evidence present in the IPED's output .csv and files.

### Para o orientador
Para usar o Llama 3.2 é preciso uma conta no Hugging Face com a licença aceita:

1. Criar uma conta no [HuggingFace](https://huggingface.co/join).
2. Aceitar os termos do [Llama 3.2 3B](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) com essa conta.
3. Gerar um token do tipo *read* em [Settings → Access Tokens](https://huggingface.co/settings/tokens).
4. Copiar `.env.example` para `.env` e preencher `HF_TOKEN=<seu token>`.

Depois, na raiz do repositório (usando GPU NVIDIA e Docker com NVIDIA Container Toolkit):
```
docker compose build
docker compose run --rm maidai                   # pipeline real (coloque as evidências em data/evidence/)
```
O modelo é baixado na primeira execução e fica no volume do Docker.

> Se rodar **fora** do Docker, use `hf auth login` no lugar do `.env`.

> Na pasta `output_iped` contém o .csv que é solicitado na demo. 
> Na pasta `./data/evidence` terá evidências "extraídas" onde algumas se relacionam de maneira intencional.