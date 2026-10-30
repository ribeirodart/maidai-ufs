"""Llama 3.2 3B via vLLM — análise forense multimodal.

Etapas: entidades -> linguagem codificada -> eventos (cruzamento áudio x imagem) ->
relacionamentos -> relatório. Gera JSONs e o relatorio_forense_final.txt.

Roda sozinho:  python src/llama_vllm.py
Lê audio.json, visual.json, text_message.json (opcional), Filelist.csv (filtrado com
pandas + regex) e contexto_filtro.txt (opcional) da pasta de evidências.
"""
import os

# Contorna o erro do nvcc (precisa ser definido ANTES de importar o vllm)
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

import gc
import json
import re
import sys
from pathlib import Path

import torch
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from evidencias import ler_csv  # noqa: E402
from relatorio import renderizar_markdown  # noqa: E402

MAX_CHARS = 12000          # limite por fonte de evidência (contexto do modelo = 32768 tokens)
MAX_CHARS_CONSOLIDADO = 40000
MAX_LINHAS_CSV = 200

# Colunas do CSV do IPED enviadas ao modelo (nome em pt-BR ou inglês, minúsculas)
COLUNAS_UTEIS_CSV = {"nome", "name", "ext", "categoria", "category", "marcador", "bookmark",
                     "caminho", "path", "deletado", "deleted", "recuperado", "carved"}

# Termos de interesse forense para filtrar o CSV (aplicado em nome/categoria/caminho; ajuste à vontade)
PADRAO_REGEX = re.compile(r"(pix|arma|droga|pagamento|entrega|palmeiras)", re.I)

REGRAS_GERAIS = """REGRAS DE ANÁLISE:
- Use SOMENTE o que está nas evidências. Não invente fatos, nomes ou locais.
- Cruze as fontes: o mesmo suspeito, local ou objeto citado no áudio e visto na imagem é UMA entidade (liste as duas fontes).
- Foque em entidades valiosas: suspeitos, locais, objetos de interesse, organizações, telefones, placas, chaves PIX.
- IGNORE números isolados sem contexto (contagens puras, durações, índices, idades). Só mantenha números que sejam
  telefone, placa, chave PIX ou valor monetário ligado a uma transação/objeto.
- Descrições de imagem que dizem "No suspicious elements visible" significam AUSÊNCIA de indício: não as use como evidência.
- Interpretações de gírias e termos codificados são HIPÓTESES: use "possível/indício", nunca afirme como fato."""


# ---------- utilidades ----------
def _ler_json(nome):
    p = config.OUTPUT_DIR / nome
    if not p.is_file():
        print(f"[Llama] Aviso: {nome} não encontrado. Usando vazio.")
        return []
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _escrever_json(dados, nome):
    p = config.OUTPUT_DIR / nome
    with open(p, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=4)
    print(f"[Llama] Salvo: {p}")


def _extrair_json(texto):
    texto = re.sub(r"```(?:json)?", "", texto).strip()
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        pass
    ini, fim = texto.find("{"), texto.rfind("}")
    if ini != -1 and fim > ini:
        try:
            return json.loads(texto[ini:fim + 1])
        except json.JSONDecodeError:
            pass
    return {"raw_output": texto}


def _compacto(obj, limite=MAX_CHARS):
    s = json.dumps(obj, ensure_ascii=False, default=str)
    if len(s) > limite:
        print(f"[Llama] Aviso: entrada cortada de {len(s)} para {limite} caracteres.")
        s = s[:limite]
    return s


def _audio_compacto(audio):
    """Remove palavras individuais (economiza tokens); mantém segmentos com timestamp."""
    return [
        {
            "arquivo": a.get("name"),
            "texto_completo": a.get("full_text"),
            "segmentos": [{"timestamp": s.get("timestamp"), "texto": s.get("text")}
                          for s in a.get("timestamps", [])],
        }
        for a in audio
    ]


def _csv_filtrado():
    if not config.FILELIST.is_file():
        print("[Llama] Filelist.csv não encontrado; seguindo sem dados do CSV.")
        return []
    try:
        df = ler_csv(config.FILELIST)
    except Exception as e:
        print(f"[Llama] Não foi possível ler o Filelist.csv: {e}")
        return []
    # Do CSV do IPED só as colunas úteis (hashes, TrackId e datas só gastariam tokens)
    uteis = [c for c in df.columns if str(c).strip().lower() in COLUNAS_UTEIS_CSV] or list(df.columns)
    filtrados = []
    for _, row in df.iterrows():
        linha = " ".join(str(row[c]) for c in uteis)
        if PADRAO_REGEX.search(linha):
            filtrados.append({str(c): str(row[c]) for c in uteis})
    print(f"[Llama] Pandas + Regex: {len(filtrados)} registro(s) relevantes no Filelist.csv.")
    return filtrados[:MAX_LINHAS_CSV]


def _contexto_externo():
    p = config.OUTPUT_DIR / "contexto_filtro.txt"
    if p.is_file():
        print("[Llama] Contexto externo carregado de contexto_filtro.txt")
        return p.read_text(encoding="utf-8").strip()
    return ""


def _filtrar_entidades(ent):
    """Remove ruído: entidades vazias ou números curtos isolados."""
    lista = ent.get("entities") if isinstance(ent, dict) else None
    if not isinstance(lista, list):
        return ent
    boas = []
    for e in lista:
        if not isinstance(e, dict):
            continue
        nome = str(e.get("nome_ou_alias", "")).strip()
        tipo = str(e.get("tipo", "")).lower()
        if not nome:
            continue
        if re.fullmatch(r"[\d\s.,:%-]{1,6}", nome) and tipo not in ("telefone", "financeiro", "placa", "veiculo"):
            continue
        boas.append(e)
    ent["entities"] = boas
    return ent


# ---------- pipeline ----------
def executar_analise_llama():
    if not config.EVIDENCE_DIR.is_dir():
        raise FileNotFoundError(f"Pasta de evidências não encontrada: {config.EVIDENCE_DIR}")

    audio = _audio_compacto(_ler_json("audio.json"))
    imagem = _ler_json("visual.json")
    mensagens = _ler_json("text_message.json")  # opcional
    csv_rel = _csv_filtrado()
    contexto = _contexto_externo() or "Nenhum"

    evidencias = (
        f"ÁUDIO (transcrição): {_compacto(audio)}\n"
        f"IMAGEM (análise visual e texto visível): {_compacto(imagem)}\n"
        f"WHATSAPP: {_compacto(mensagens)}\n"
        f"CSV FILTRADO: {_compacto(csv_rel)}\n"
        f"CONTEXTO ADICIONAL: {contexto}"
    )

    print("[Llama] Carregando modelo na VRAM (vLLM)...")
    llm = LLM(model="meta-llama/Llama-3.2-3B-Instruct", max_model_len=32768, gpu_memory_utilization=0.8)
    # Temperatura 0 = decodificação determinística, ideal para extração de JSON
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=4096)

    def gerar(system, user):
        dados = {}
        for _ in range(2):  # tenta de novo uma vez se o JSON vier inválido
            saidas = llm.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                sampling_params,
            )
            dados = _extrair_json(saidas[0].outputs[0].text)
            if "raw_output" not in dados:
                return dados
        return dados

    # ---- BLOCO 1: entidades ----
    print("[Llama] Bloco 1/5: entidades")
    prompt_ent = f"""Analise as evidências digitais e liste as entidades valiosas para a investigação.

{REGRAS_GERAIS}

{evidencias}

Retorne APENAS um JSON válido neste formato:
{{"entities": [{{"entity_id": "ENT_001", "nome_ou_alias": "...", "tipo": "suspeito/pessoa/local/objeto_de_interesse/organizacao/telefone/financeiro/veiculo/documento", "fontes": ["AUDIO", "IMAGEM"], "arquivos": ["nome do arquivo"], "locais_na_fonte": ["timestamp ou índice"], "relevancia": "alta/media"}}]}}"""
    entidades = _filtrar_entidades(gerar(
        "Você extrai entidades de evidências digitais para um perito. Responda apenas JSON válido.", prompt_ent))
    _escrever_json(entidades, "entities.json")

    # ---- BLOCO 2: linguagem codificada (cruzamento multimodal) ----
    print("[Llama] Bloco 2/5: linguagem codificada")
    prompt_cod = f"""Identifique gírias e termos CODIFICADOS que, no contexto, possam ter outro significado investigativo.
Exemplo de raciocínio: se um áudio fala em distribuir "café" ou "caixas" em comunidades/favelas, junto de pagamentos,
quantidades, entregas ou cautela ao falar, o termo pode ser código para entorpecentes e NÃO o insumo literal.
Cruze com as imagens (ex.: embalagens, dinheiro, armas) e com o CSV para reforçar ou enfraquecer a hipótese.
Só gere um alerta quando houver indícios de contexto; se o termo está sendo usado de forma ordinária, NÃO alerte.

{REGRAS_GERAIS}

{evidencias}

Retorne APENAS um JSON válido neste formato (lista vazia se não houver alertas):
{{"alertas": [{{"termo": "termo usado", "significado_provavel": "hipótese investigativa", "contexto": "trecho que sustenta", "arquivo": "nome do arquivo", "timestamp": "[mm:ss -> mm:ss] ou null", "confianca": "baixa/media/alta", "justificativa": "indícios de contexto e cruzamento com outras fontes"}}]}}"""
    alertas = gerar("Você é um analista de inteligência especializado em linguagem codificada. Responda apenas JSON válido.", prompt_cod)
    _escrever_json(alertas, "linguagem_codificada.json")

    # ---- BLOCO 3: eventos ----
    print("[Llama] Bloco 3/5: eventos")
    prompt_evt = f"""Extraia os EVENTOS relevantes, com atenção a eventos ligados a crimes, correlacionando áudio e imagem.

Tipos de evento (apenas guia; NÃO use se não estiver na evidência):
transacao (pagamento, pix, cobrança), negociacao, entrega, encontro, ameaca, porte_arma, trafico_drogas,
ocultacao, comunicacao.

ENTIDADES JÁ IDENTIFICADAS: {_compacto(entidades)}
ALERTAS DE LINGUAGEM CODIFICADA: {_compacto(alertas)}

{REGRAS_GERAIS}

{evidencias}

Retorne APENAS um JSON válido neste formato:
{{"eventos": [{{"event_id": "EVT_001", "tipo": "entrega", "descricao": "descrição em português suportada pela evidência", "entidades": ["ENT_001"], "fonte": "AUDIO/IMAGEM/WHATSAPP/CSV", "arquivo": "nome do arquivo", "timestamp": "[mm:ss -> mm:ss] ou null", "evidencias_cruzadas": ["outros arquivos que reforçam"], "interpretacao": "hipótese investigativa ou null"}}]}}"""
    eventos = gerar("Você é um analista forense. Extraia eventos com rigor, sem inventar. Responda apenas JSON válido.", prompt_evt)
    _escrever_json(eventos, "events.json")

    # ---- BLOCO 4: relacionamentos ----
    print("[Llama] Bloco 4/5: relacionamentos")
    prompt_rel = f"""Relacione cada evento às entidades que participam dele (rede de relacionamentos).

ENTIDADES: {_compacto(entidades)}
EVENTOS: {_compacto(eventos)}

Retorne APENAS um JSON válido neste formato:
{{"entity_event_relationships": [{{"entity_id": "ENT_001", "event_id": "EVT_001", "papel": "remetente/participante/destinatário/local/objeto", "arquivo": "nome do arquivo", "timestamp": "..."}}]}}"""
    relacoes = gerar("Você relaciona entidades e eventos de uma investigação. Responda apenas JSON válido.", prompt_rel)
    _escrever_json(relacoes, "entity_event_relationships.json")

    consolidado = {"entidades": entidades, "eventos": eventos, "relacionamentos": relacoes,
                   "linguagem_codificada": alertas}
    _escrever_json(consolidado, "consolidado.json")

    # ---- BLOCO 5: relatório ----
    print("[Llama] Bloco 5/5: relatório forense")
    prompt_final = f"""Com base no JSON consolidado, elabore o conteúdo de um relatório forense em PORTUGUÊS.
Priorize suspeitos, locais, objetos de interesse e organizações; não destaque números isolados sem contexto.
Trate interpretações de linguagem codificada como hipóteses. Use somente o que está no JSON.

JSON: {_compacto(consolidado, MAX_CHARS_CONSOLIDADO)}

Retorne APENAS um JSON válido neste formato:
{{"relatorio_forense": {{
  "resumo_executivo": "síntese objetiva dos fatos constatados e das principais hipóteses",
  "achados": [{{"evidencia": "nome do arquivo", "fonte": "AUDIO/IMAGEM/WHATSAPP/CSV", "timestamp": "[mm:ss -> mm:ss] ou null", "tipo": "tipo do evento", "descricao": "descrição em português"}}],
  "pontos_atencao": ["..."],
  "conclusao_preliminar": "conclusão estritamente baseada nas evidências"
}}}}"""
    relatorio = gerar("Você é um perito forense digital sênior. Escreva em português. Responda apenas JSON válido.", prompt_final)
    _escrever_json(relatorio, "forensic_summary.json")

    # ---- relatório em texto/Markdown ----
    texto = renderizar_markdown(relatorio, consolidado, alertas)
    destino = config.OUTPUT_DIR / "relatorio_forense_final.txt"
    destino.write_text(texto, encoding="utf-8")
    print(f"[Llama] Relatório formatado salvo em: {destino}")

    del llm
    gc.collect()
    try:
        from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass
    torch.cuda.empty_cache()
    print("[Llama] Modelo descarregado.")
    return relatorio


if __name__ == "__main__":
    executar_analise_llama()
