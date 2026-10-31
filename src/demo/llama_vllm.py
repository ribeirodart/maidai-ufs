import os

os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

import gc
import json
import re
import sys
import unicodedata
from pathlib import Path

import torch
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from evidencias import ler_csv  # noqa: E402
from evidencias import ler_txt  # noqa: E402
from evidencias import _parse_mensagens  # noqa: E402
from relatorio import renderizar_markdown  # noqa: E402

DEBUG = os.environ.get("MAIDAI_DEBUG") == "1"

MAX_CHARS = 12000
MAX_CHARS_CONSOLIDADO = 40000
MAX_LINHAS_CSV = 200
MAX_ARQUIVOS_GEO_NO_PROMPT = 50

COLUNAS_UTEIS_CSV = {"nome", "name", "ext", "categoria", "category", "marcador", "bookmark",
                     "caminho", "path", "deletado", "deleted", "recuperado", "carved"}

COL_GEO = re.compile(r"(?<![a-z])(lat(itude)?|lon(g(itude)?)?|alt(itude)?)(?![a-z])", re.I)

PADRAO_REGEX = re.compile(r"(pix|arma|droga|pagamento|entrega|palmeiras)", re.I)

TIPOS_EVENTO = ("entrega", "transacao", "negociacao", "encontro", "comunicacao", "ameaca",
                "porte_arma", "trafico_drogas", "ocultacao", "outro")
RELACOES_EVENTO_EVENTO = ("antes", "depois", "ao_mesmo_tempo", "porque", "como_resultado")
RELACOES_CAUSAIS = ("porque", "como_resultado")
TIPOS_ENTIDADE = ("pessoa", "apelido", "local", "objeto_de_interesse", "organizacao", "telefone", "financeiro", "veiculo")
TIPOS_PESSOA = ("pessoa", "apelido")
TIPOS_COM_RELACAO = ("pessoa", "apelido", "organizacao", "local", "veiculo", "telefone")
PAPEIS = ("remetente", "destinatario", "pagador", "recebedor", "local", "objeto", "participante")
TIPOS_NUMERICOS_OK = ("telefone", "financeiro", "placa", "veiculo", "valor")
FONTES_FALA = ("AUDIO", "WHATSAPP")
LIMIAR_TRECHO = 0.75
LIMITE_TRECHO = 400

REGEX_DATA_HORA = re.compile(r"^\W*(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{1,2}:\d{2}(:\d{2})?)\W*$")
RE_DATA = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b")
RE_HORA = re.compile(r"\b(\d{1,2}:\d{2}(?::\d{2})?)\b")
RE_VALOR = re.compile(r"R\$\s?\d[\d.]*(?:,\d{2})?")
RE_TRANSACAO = re.compile(r"\b(pix|transfer[eê]ncia|pagamento|comprovante)\b", re.I)
RE_INICIO_AUDIO = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)")
MARCAS_DOC = re.compile(r"simulad|fict[ií]ci|sint[eé]tic|amostra|sem valor|sem validade|exemplo|demonstra", re.I)

REGRAS_GERAIS = """DIRETRIZES DE ANÁLISE FORENSE (LEAP):
- FOCO RELACIONAL E SUSPEITA: não julgue moralmente nem decrete culpa; avalie se as relações entre entidades e eventos configuram indícios de comportamento suspeito, com base estrita nas evidências.
- USE SOMENTE AS EVIDÊNCIAS: não invente fatos, nomes, intenções, arquivos ou locais. Copie nomes e termos exatamente como aparecem.
- CRUZAMENTO: o mesmo suspeito, local ou objeto citado no áudio e visto na imagem é UMA única entidade.
- RIGOR EM EVENTOS: agendamentos, encontros, transações e entregas registram O QUE, ONDE, QUANDO e QUEM, com os termos literais da evidência.
- HIPÓTESES E LACUNAS: gírias e termos codificados são hipóteses ("possível indício"). Ausência de elemento não é evento.
- IMAGEM: descrição que diz "No suspicious elements visible" significa ausência de indício, não evidência.
- Os formatos de saída abaixo são MOLDES: nunca copie o texto entre < > nem inclua nada que não esteja nas evidências.
- Se não houver nada a registrar, devolva a lista vazia."""

FORMATO_ENT = ('{"entities": [{"nome": "<nome exato como aparece na evidência>", '
               '"tipo": "pessoa|apelido|local|objeto_de_interesse|organizacao|telefone|financeiro|veiculo"}]}')

FORMATO_COD = ('{"alertas": [{"termo": "<palavra ou expressão curta, dita literalmente>", '
               '"significado_provavel": "<hipótese investigativa>", '
               '"justificativa": "<indício de contexto que sustenta a hipótese>", '
               '"confianca": "baixa|media|alta"}]}')

FORMATO_EVT = ('{"eventos": [{"tipo": "entrega|transacao|negociacao|encontro|comunicacao|ameaca|porte_arma|trafico_drogas|ocultacao|outro", '
               '"descricao": "<UMA frase autocontida: o quê, quem, onde, quando>", '
               '"trecho": "<citação literal da evidência, 3 a 15 palavras>", '
               '"participantes": [{"nome": "<entidade da lista>", "papel": "remetente|destinatario|entregador|pagador|recebedor|local|objeto|participante"}], '
               '"conceitos": ["<conceito abstrato 1-2 palavras>", "<conceito 2>"], '
               '"analise_relacionamento_suspeito": "<por que a relação interessa à investigação, só com base no trecho; ou null>"}]}')

FORMATO_REL_EE = '{"relacoes": [{"origem": "<entidade da lista>", "relacao": "<verbo>", "destino": "<entidade da lista>"}]}'
FORMATO_REL_VV = '{"relacoes": [{"origem": "<EVT_xxx>", "relacao": "porque|como_resultado", "destino": "<EVT_xxx>"}]}'
FORMATO_REL = ('{"relatorio_forense": {"resumo_executivo": "<fatos constatados e principais hipóteses>", '
               '"pontos_atencao": ["<ponto>"], "conclusao_preliminar": "<conclusão estritamente baseada nas evidências>"}}')


def _prompt_entidades(ev):
    return f"""Extraia as ENTIDADES das evidências: substantivos específicos (pessoas, apelidos, locais, objetos de interesse, organizações, telefones, valores, veículos). Sem pronomes. NÃO liste: datas, horários, números soltos, nomes de campos de formulário (como Pagador, Valor, Data), meios de pagamento genéricos, marcações do documento (como simulado ou amostra) nem a descrição da cena.

{REGRAS_GERAIS}

{ev}

Retorne APENAS um JSON válido neste molde:
{FORMATO_ENT}"""


def _prompt_codificada(ev):
    return f"""Identifique TERMOS CODIFICADOS: palavras comuns usadas em ÁUDIO ou WHATSAPP com possível sentido diferente do literal (gíria ou código), inclusive palavras vagas que substituem um objeto ou ação (como 'coisa', 'material', 'encomenda', 'pacote'). Alerte quando o CONTEXTO sugerir atividade ilícita ou cautela (pagamento, quantidade, entrega, hora e local combinados para buscar algo); se o indício for fraco, use confiança baixa. Uso ordinário da palavra, sem esse contexto, não é alerta. Não liste nomes de pessoas, locais, bancos, documentos, valores nem conteúdo de imagens.

{REGRAS_GERAIS}

{ev}

Retorne APENAS um JSON válido neste molde:
{FORMATO_COD}"""


def _prompt_eventos(ev, entidades, alertas):
    lista = "; ".join(f"{e['entity_id']}={e['nome_ou_alias']}" for e in entidades) or "nenhuma"
    cods = "; ".join(f"{a['termo']} -> {a['significado_provavel']}" for a in alertas) or "nenhum"
    return f"""Extraia os EVENTOS (AutoSchemaKG: entidade-evento). Cada evento é UMA frase autocontida, ancorada em um TRECHO copiado literalmente da evidência. Cruze áudio e imagem para indicar quem se relaciona, onde e por quê. A transcrição de áudio não identifica quem fala, então não atribua 'remetente' sem o nome constar no trecho. A descricao é uma frase com sujeito, ação e objeto, incluindo valor, data e horário quando constarem; não repita só o título do documento.
Use como participantes apenas entidades da lista que constem na MESMA fonte (a mesma conversa, o mesmo áudio ou a mesma imagem). Se o trecho usar pronome ('ele', 'ela') ou referência vaga, resolva-o pelas mensagens ANTERIORES e escreva o NOME na descricao e em participantes. Em entregas e encontros, a descricao diz quem leva, o quê, para quem, onde e a que horas, e participantes inclui quem leva (papel entregador), quem recebe e o local.

ENTIDADES: {lista}
TERMOS CODIFICADOS (hipóteses): {cods}

{REGRAS_GERAIS}

{ev}

Retorne APENAS um JSON válido neste molde:
{FORMATO_EVT}"""


def _prompt_rel_entidades(entidades, eventos):
    lista = "; ".join(e["nome_ou_alias"] for e in entidades)
    evts = "\n".join(f"- {v['event_id']}: {v['descricao']}" for v in eventos)
    return f"""Liste as RELAÇÕES ENTRE ENTIDADES (AutoSchemaKG: entidade-entidade). A relação é um verbo conjugado. Relacione apenas pessoas, organizações, locais e veículos (não identificadores, datas, valores nem meios de pagamento). Use apenas entidades da lista e só relações sustentadas pelos eventos abaixo.

ENTIDADES: {lista}
EVENTOS:
{evts}

Retorne APENAS um JSON válido neste molde:
{FORMATO_REL_EE}"""


def _prompt_rel_eventos(eventos):
    evts = "\n".join(f"- {v['event_id']}: {v['descricao']} ({v.get('timestamp') or 'sem timestamp'})" for v in eventos)
    return f"""Liste as RELAÇÕES CAUSAIS ENTRE EVENTOS (AutoSchemaKG: evento-evento): porque ou como_resultado. A ordem no tempo o sistema já calcula; não a informe. Use apenas os ids abaixo e só relações que as descrições afirmam de forma explícita; se não houver, devolva a lista vazia.

EVENTOS:
{evts}

Retorne APENAS um JSON válido neste molde:
{FORMATO_REL_VV}"""


def _prompt_relatorio(resumo):
    return f"""Com base no JSON consolidado, escreva o conteúdo do relatório em PORTUGUÊS. Priorize pessoas citadas, locais, objetos de interesse e organizações; só chame alguém de 'suspeito' se os eventos trouxerem indício explícito de ilícito. Trate interpretações de linguagem codificada como hipóteses. Se 'marcas_de_documento' indicar material simulado ou fictício, diga isso no resumo. Cite SOMENTE nomes, arquivos e fatos presentes no JSON; se houver poucos elementos, diga isso.

JSON CONSOLIDADO: {resumo}

Retorne APENAS um JSON válido neste molde:
{FORMATO_REL}"""


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


def _itens(dados, chave):
    """Lista de dicts de dados[chave], tolerando lista solta ou saída inválida."""
    v = dados.get(chave) if isinstance(dados, dict) else dados
    return [x for x in v if isinstance(x, dict)] if isinstance(v, list) else []


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def _csv_filtrado():
    if not config.FILELIST.is_file():
        return []
    try:
        df = ler_csv(config.FILELIST)
    except Exception:
        return []
    uteis = [c for c in df.columns if str(c).strip().lower() in COLUNAS_UTEIS_CSV] or list(df.columns)
    filtrados = []
    for _, row in df.iterrows():
        linha = " ".join(str(row[c]) for c in uteis)
        if PADRAO_REGEX.search(linha):
            filtrados.append({str(c): str(row[c]) for c in uteis})
    return filtrados[:MAX_LINHAS_CSV]


def _geolocalizacao():
    if not config.FILELIST.is_file():
        return []
    try:
        df = ler_csv(config.FILELIST)
    except Exception:
        return []
    cols_geo = [c for c in df.columns if COL_GEO.search(str(c))]
    if not cols_geo:
        return []
    col_nome = next((c for c in df.columns if str(c).strip().lower() in ("nome", "name")), None)
    achados = []
    for _, row in df.iterrows():
        geo = {str(c): str(row[c]).strip() for c in cols_geo if str(row[c]).strip() not in ("", "nan", "None")}
        if geo:
            achados.append({"arquivo": str(row[col_nome]) if col_nome is not None else "?", **geo})
    return achados


def _secao_geo(geos):
    if not geos:
        return ""
    linhas = ["", "## Metadados de geolocalização (verificar no IPED)", "",
              "Os arquivos abaixo possuem coordenadas nos metadados. O local NÃO foi interpretado: "
              "recomenda-se que o perito confira a posição de cada um no IPED (SearchApp).", ""]
    for g in geos:
        coords = ", ".join(f"{k}: {v}" for k, v in g.items() if k != "arquivo")
        linhas.append(f"- **{g['arquivo']}** — {coords}")
    return "\n".join(linhas) + "\n"


def _contexto_externo():
    p = config.OUTPUT_DIR / "contexto_filtro.txt"
    return p.read_text(encoding="utf-8").strip() if p.is_file() else ""


def _montar_trechos(audio, imagem, mensagens):
    """Índice dos trechos citáveis (falas, descrições, textos visíveis e mensagens)."""
    trechos = []

    def add(fonte, arquivo, local, texto):
        texto = str(texto or "").strip()
        if texto and texto.lower() not in ("nan", "none"):
            n = _norm(texto)
            trechos.append({"i": len(trechos), "fonte": fonte, "arquivo": str(arquivo), "local": local,
                            "texto": texto, "toks": set(n.split())})

    for a in audio:
        segs = a.get("timestamps") or []
        for s in segs:
            add("AUDIO", a.get("name"), s.get("timestamp"), s.get("text"))
        if not segs:
            add("AUDIO", a.get("name"), None, a.get("full_text"))
    for im in imagem:
        add("IMAGEM", im.get("name"), "descrição", im.get("description"))
        add("IMAGEM", im.get("name"), "texto visível", im.get("visible_text"))
    for m in mensagens if isinstance(mensagens, list) else []:
        if isinstance(m, dict):
            texto = m.get("mensagem") or m.get("message") or m.get("text") or m.get("texto")
            if texto and m.get("remetente"):
                texto = f"{m['remetente']}: {texto}"
            add("WHATSAPP", "text_message.txt", m.get("data_hora") or m.get("index"), texto)
    return trechos


def _cortar_linhas(linhas, limite, rotulo):
    """Corta em limite de caracteres sem partir uma linha ao meio."""
    saida, total = [], 0
    for i, linha in enumerate(linhas):
        if total + len(linha) + 1 > limite:
            print(f"[Llama] Aviso: {rotulo} cortado ({i}/{len(linhas)} linhas) por exceder {limite} caracteres.")
            saida.append("[... evidência cortada por limite de tamanho ...]")
            break
        saida.append(linha)
        total += len(linha) + 1
    return "\n".join(saida) or "(nenhum)"


def _texto_evidencias(audio, imagem, mensagens, csv_rel, geos, contexto):
    l_audio = []
    for a in audio:
        l_audio.append(f"=== ÁUDIO: {a.get('name')} ===")
        segs = a.get("timestamps") or []
        l_audio += [f"{s.get('timestamp')} {str(s.get('text', '')).strip()}" for s in segs] \
            or [str(a.get("full_text", ""))]
    l_img = []
    for im in imagem:
        linha = f"=== IMAGEM: {im.get('name')} === {str(im.get('description', '')).strip()}"
        if im.get("visible_text"):
            linha += f" | TEXTO VISÍVEL: {str(im['visible_text']).strip()}"
        l_img.append(linha)
    l_msg = [f"[{m.get('data_hora', '?')}] {m.get('remetente', '?')}: {m.get('mensagem', '')}"
             for m in (mensagens if isinstance(mensagens, list) else []) if isinstance(m, dict)]
    l_csv = [" | ".join(r.values()) for r in csv_rel]
    arquivos_geo = ", ".join(g["arquivo"] for g in geos[:MAX_ARQUIVOS_GEO_NO_PROMPT]) or "nenhum"
    return (f"ÁUDIO (transcrição com timestamps):\n{_cortar_linhas(l_audio, MAX_CHARS, 'ÁUDIO')}\n\n"
            f"IMAGEM (análise visual):\n{_cortar_linhas(l_img, MAX_CHARS, 'IMAGEM')}\n\n"
            f"WHATSAPP:\n{_cortar_linhas(l_msg, MAX_CHARS, 'WHATSAPP')}\n\n"
            f"CSV FILTRADO:\n{_cortar_linhas(l_csv, MAX_CHARS, 'CSV')}\n\n"
            f"ARQUIVOS COM GPS: {arquivos_geo}\n"
            f"CONTEXTO ADICIONAL: {contexto}")


def _proveniencia(trechos, nomes):
    """Trechos em que aparecem todas as palavras de pelo menos um dos nomes."""
    alvos = [set(_norm(n).split()) for n in nomes if _norm(n)]
    return [t for t in trechos if any(a <= t["toks"] for a in alvos)]


def _melhor_trecho(trechos, citacao):
    """Trecho que melhor contém a citação; None se menos de LIMIAR_TRECHO das palavras constam."""
    q = [w for w in _norm(citacao).split() if len(w) >= 3]
    if len(q) < 3:
        return None
    melhor, cobertura = None, 0.0
    for t in trechos:
        c = sum(1 for w in q if w in t["toks"]) / len(q)
        if c > cobertura:
            melhor, cobertura = t, c
    return melhor if cobertura >= LIMIAR_TRECHO else None


def _slug(s):
    return _norm(s).replace(" ", "_")


def _gerar_lote(llm, params, pedidos):
    """Gera as respostas em lote. Se o JSON vier inválido, repete a pergunta apontando o erro."""
    conversas = [[{"role": "system", "content": s}, {"role": "user", "content": u}] for s, u in pedidos]
    resultados = [{} for _ in pedidos]
    pendentes = list(range(len(pedidos)))
    for _ in range(2):
        if not pendentes:
            break
        saidas = llm.chat([conversas[i] for i in pendentes], params)
        novos = []
        for i, saida in zip(pendentes, saidas):
            texto = saida.outputs[0].text
            resultados[i] = _extrair_json(texto)
            if isinstance(resultados[i], dict) and "raw_output" in resultados[i]:
                conversas[i] += [{"role": "assistant", "content": texto},
                                 {"role": "user", "content": "Sua resposta não é um JSON válido. "
                                  "Responda SOMENTE com o JSON no molde pedido, sem texto extra."}]
                novos.append(i)
        pendentes = novos
    return resultados


def _validar_entidades(proposta, trechos):
    vistos = {}
    for e in _itens(proposta, "entities"):
        nome = str(e.get("nome") or e.get("nome_ou_alias") or "").strip()
        tipo = _slug(e.get("tipo", "")) or "outro"
        if tipo not in TIPOS_ENTIDADE:
            continue
        chave = _norm(nome)
        if not chave or len(nome) > 80 or chave in vistos:
            continue
        if re.fullmatch(r"[\d\s.,:%-]{1,6}", nome) and tipo not in TIPOS_NUMERICOS_OK:
            continue
        vistos[chave] = {"nome": nome, "tipo": tipo}

    pessoas = {k: v for k, v in vistos.items() if v["tipo"] in ("pessoa", "apelido", "suspeito")}
    aliases, absorvidos = {}, set()
    for k in pessoas:
        donos = [k2 for k2 in pessoas if k2 != k and set(k.split()) < set(k2.split())]
        if len(donos) == 1:
            aliases.setdefault(donos[0], []).append(vistos[k]["nome"])
            absorvidos.add(k)

    entidades, prov = [], {}
    for k, v in vistos.items():
        if k in absorvidos:
            continue
        achados = _proveniencia(trechos, [v["nome"]] + aliases.get(k, []))
        if not achados:
            continue
        eid = f"ENT_{len(entidades) + 1:03d}"
        arquivos = sorted({t["arquivo"] for t in achados})
        ent = {"entity_id": eid, "nome_ou_alias": v["nome"], "tipo": v["tipo"],
               "fontes": sorted({t["fonte"] for t in achados}), "arquivos": arquivos,
               "locais_na_fonte": list(dict.fromkeys(f"{t['arquivo']} {t['local'] or ''}".strip() for t in achados))[:5],
               "relevancia": "alta" if len(arquivos) >= 2 else "media"}
        if k in aliases:
            ent["aliases"] = aliases[k]
        entidades.append(ent)
        prov[eid] = {t["i"] for t in achados}
    return entidades, prov, len(vistos) - len(entidades) - len(absorvidos)


def _validar_alertas(proposta, trechos, entidades):
    nomes_ent = {_norm(e["nome_ou_alias"]) for e in entidades}
    nomes_ent |= {_norm(a) for e in entidades for a in e.get("aliases", [])}
    fala = [t for t in trechos if t["fonte"] in FONTES_FALA]
    alertas, vistos, descartados = [], set(), 0
    for a in _itens(proposta, "alertas"):
        termo = str(a.get("termo", "")).strip()
        n = _norm(termo)
        sig = str(a.get("significado_provavel", "")).strip()
        if not n or not sig or len(n.split()) > 4 or n in vistos or n in nomes_ent:
            descartados += 1
            continue
        achados = _proveniencia(fala, [termo])
        if not achados:
            descartados += 1
            continue
        vistos.add(n)
        conf = _slug(a.get("confianca", ""))
        conf = conf if conf in ("baixa", "media", "alta") else "baixa"
        if conf == "alta" and len({(t["arquivo"], t["local"]) for t in achados}) < 2:
            conf = "media"
        t0 = achados[0]
        alertas.append({"termo": termo, "significado_provavel": sig, "contexto": t0["texto"][:240],
                        "arquivo": t0["arquivo"], "timestamp": t0["local"], "confianca": conf,
                        "justificativa": str(a.get("justificativa", "")).strip()})
    return alertas, descartados


def _validar_eventos(proposta, trechos, entidades):
    idx = {}
    for e in entidades:
        for nome in [e["nome_ou_alias"]] + e.get("aliases", []):
            idx[_norm(nome)] = e["entity_id"]
    arquivos_ent = {e["entity_id"]: set(e["arquivos"]) for e in entidades}
    eventos, vistos, descartados = [], set(), 0
    for ev in _itens(proposta, "eventos"):
        desc = str(ev.get("descricao", "")).strip()
        t = _melhor_trecho(trechos, ev.get("trecho", ""))
        tipo = _slug(ev.get("tipo", ""))
        tipo = tipo if tipo in TIPOS_EVENTO else "outro"
        if not desc or t is None or (t["i"], tipo) in vistos:
            descartados += 1
            continue
        vistos.add((t["i"], tipo))
        parts = []
        for p in ev.get("participantes") or []:
            nome = p.get("nome") if isinstance(p, dict) else p
            eid = idx.get(_norm(nome))
            if eid and eid not in {x["entity_id"] for x in parts}:
                papel = str(p.get("papel", "")).strip() if isinstance(p, dict) else ""
                parts.append({"entity_id": eid, "papel": papel or "participante"})
        cruzadas = sorted(set().union(*[arquivos_ent[x["entity_id"]] for x in parts]) - {t["arquivo"]}) if parts else []
        conceitos = [str(c).strip() for c in (ev.get("conceitos") or []) if str(c).strip()][:3]
        analise = ev.get("analise_relacionamento_suspeito")
        eventos.append({
            "event_id": f"EVT_{len(eventos) + 1:03d}", "tipo": tipo, "descricao": desc,
            "entidades": [x["entity_id"] for x in parts], "participantes": parts,
            "fonte": t["fonte"], "arquivo": t["arquivo"],
            "timestamp": t["local"] if t["fonte"] in FONTES_FALA else None,
            "trecho_literal": t["texto"][:240], "evidencias_cruzadas": cruzadas, "conceitos": conceitos,
            "analise_relacionamento_suspeito": str(analise).strip() if analise and str(analise).lower() != "null" else None,
        })
    return eventos, descartados


def _validar_relacoes(prop_ee, prop_vv, entidades, eventos, prov):
    ent_evt = [{"entity_id": p["entity_id"], "event_id": v["event_id"], "papel": p["papel"],
                "arquivo": v["arquivo"], "timestamp": v["timestamp"]}
               for v in eventos for p in v["participantes"]]

    por_nome = {_norm(e["nome_ou_alias"]): e for e in entidades}
    por_nome.update({_norm(a): e for e in entidades for a in e.get("aliases", [])})
    juntos_em_evento = [set(v["entidades"]) for v in eventos]

    ent_ent, vistos = [], set()
    for r in _itens(prop_ee, "relacoes"):
        a, b = por_nome.get(_norm(r.get("origem", ""))), por_nome.get(_norm(r.get("destino", "")))
        rel = str(r.get("relacao", "")).strip()[:40]
        if not (a and b and rel) or a is b:
            continue
        ia, ib = a["entity_id"], b["entity_id"]
        coocorrem = any({ia, ib} <= s for s in juntos_em_evento) or bool(prov.get(ia, set()) & prov.get(ib, set()))
        if not coocorrem or (ia, rel, ib) in vistos:
            continue
        vistos.add((ia, rel, ib))
        ent_ent.append({"origem_id": ia, "origem": a["nome_ou_alias"], "relacao": rel,
                        "destino_id": ib, "destino": b["nome_ou_alias"]})

    ids = {v["event_id"] for v in eventos}
    evt_evt, vistos = [], set()
    for r in _itens(prop_vv, "relacoes"):
        a, b, rel = str(r.get("origem", "")).strip(), str(r.get("destino", "")).strip(), _slug(r.get("relacao", ""))
        if a in ids and b in ids and a != b and rel in RELACOES_EVENTO_EVENTO and (a, rel, b) not in vistos:
            vistos.add((a, rel, b))
            evt_evt.append({"origem_id": a, "relacao": rel, "destino_id": b})
    return {"entity_event_relationships": ent_evt, "entity_entity_relationships": ent_ent,
            "event_event_relationships": evt_evt}


def _secao_grafo(relacoes, eventos):
    linhas = []
    if relacoes["entity_entity_relationships"]:
        linhas += ["", "## Relações entre entidades", ""]
        linhas += [f"- {r['origem']} → {r['relacao']} → {r['destino']}" for r in relacoes["entity_entity_relationships"]]
    if relacoes["event_event_relationships"]:
        desc = {v["event_id"]: v["descricao"][:80] for v in eventos}
        linhas += ["", "## Sequência e causalidade entre eventos", ""]
        linhas += [f"- {r['origem_id']} ({desc.get(r['origem_id'], '')}) → {r['relacao'].replace('_', ' ')} → "
                   f"{r['destino_id']} ({desc.get(r['destino_id'], '')})" for r in relacoes["event_event_relationships"]]
    return "\n".join(linhas) + "\n" if linhas else ""


def _inserir_antes_da_nota(base, extra):
    if not extra.strip():
        return base
    corte = base.rfind("\n---\n")
    if corte == -1:
        return base.rstrip("\n") + "\n\n" + extra.lstrip("\n")
    return base[:corte].rstrip("\n") + "\n\n" + extra.strip("\n") + "\n" + base[corte:]


def _debug(titulo, dados):
    if DEBUG:
        print(f"\n[DEBUG - {titulo}] {json.dumps(dados, ensure_ascii=False)[:600]}\n")


def executar_analise_llama():
    if not config.EVIDENCE_DIR.is_dir():
        raise FileNotFoundError(f"Pasta de evidências não encontrada: {config.EVIDENCE_DIR}")

    audio, imagem, mensagens = _ler_json("audio.json"), _ler_json("visual.json"), _parse_mensagens(ler_txt())
    trechos = _montar_trechos(audio, imagem, mensagens)
    if not trechos:
        print("[Llama] Nenhuma evidência em audio.json, visual.json ou text_message.txt; análise cancelada.")
        return None
    csv_rel, geos = _csv_filtrado(), _geolocalizacao()
    evidencias = _texto_evidencias(audio, imagem, mensagens, csv_rel, geos, _contexto_externo() or "Nenhum")
    SISTEMA = "Você é um perito forense digital. Responda apenas JSON válido, em português."

    print("[Llama] Carregando modelo na VRAM (vLLM)...")
    llm = LLM(model="meta-llama/Llama-3.2-3B-Instruct", max_model_len=32768, gpu_memory_utilization=0.8)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=4096)

    try:
        print("[Llama] Etapas 1-2/5: entidades e linguagem codificada")
        prop_ent, prop_cod = _gerar_lote(llm, sampling_params, [
            (SISTEMA, _prompt_entidades(evidencias)), (SISTEMA, _prompt_codificada(evidencias))])
        entidades, prov, desc_ent = _validar_entidades(prop_ent, trechos)
        alertas, desc_cod = _validar_alertas(prop_cod, trechos, entidades)
        print(f"[Llama] Entidades: {len(entidades)} confirmadas na evidência ({desc_ent} descartadas) | "
              f"alertas: {len(alertas)} ({desc_cod} descartados)")
        _escrever_json({"entities": entidades}, "entities.json")
        _escrever_json({"alertas": alertas}, "linguagem_codificada.json")
        _debug("Entidades", entidades)
        _debug("Linguagem codificada", alertas)

        print("[Llama] Etapa 3/5: eventos")
        prop_evt, = _gerar_lote(llm, sampling_params, [(SISTEMA, _prompt_eventos(evidencias, entidades, alertas))])
        eventos, desc_evt = _validar_eventos(prop_evt, trechos, entidades)
        print(f"[Llama] Eventos: {len(eventos)} ancorados na evidência ({desc_evt} descartados)")
        _escrever_json({"eventos": eventos}, "events.json")
        _debug("Eventos", eventos)

        print("[Llama] Etapa 4/5: relações")
        pedidos = []
        if len(entidades) >= 2 and eventos:
            pedidos.append((SISTEMA, _prompt_rel_entidades(entidades, eventos)))
        if len(eventos) >= 2:
            pedidos.append((SISTEMA, _prompt_rel_eventos(eventos)))
        resp = iter(_gerar_lote(llm, sampling_params, pedidos)) if pedidos else iter(())
        prop_ee = next(resp) if len(entidades) >= 2 and eventos else {}
        prop_vv = next(resp) if len(eventos) >= 2 else {}
        relacoes = _validar_relacoes(prop_ee, prop_vv, entidades, eventos, prov)
        _escrever_json(relacoes, "entity_event_relationships.json")
        _debug("Relações", relacoes)

        consolidado = {"entidades": {"entities": entidades}, "eventos": {"eventos": eventos},
                       "relacionamentos": relacoes, "linguagem_codificada": {"alertas": alertas},
                       "validacao": {"entidades_descartadas": desc_ent, "alertas_descartados": desc_cod,
                                     "eventos_descartados": desc_evt}}
        _escrever_json(consolidado, "consolidado.json")

        print("[Llama] Etapa 5/5: relatório forense")
        if eventos or alertas:
            resumo = json.dumps({
                "entidades": [{"nome": e["nome_ou_alias"], "tipo": e["tipo"], "arquivos": e["arquivos"]} for e in entidades],
                "eventos": [{"id": v["event_id"], "tipo": v["tipo"], "descricao": v["descricao"], "arquivo": v["arquivo"],
                             "timestamp": v["timestamp"], "analise": v["analise_relacionamento_suspeito"]} for v in eventos],
                "linguagem_codificada": [{"termo": a["termo"], "hipotese": a["significado_provavel"],
                                          "confianca": a["confianca"]} for a in alertas],
                "relacoes_entre_entidades": relacoes["entity_entity_relationships"],
                "relacoes_entre_eventos": relacoes["event_event_relationships"],
            }, ensure_ascii=False)[:MAX_CHARS_CONSOLIDADO]
            relatorio, = _gerar_lote(llm, sampling_params, [(SISTEMA, _prompt_relatorio(resumo))])
        else:
            relatorio = {"relatorio_forense": {
                "resumo_executivo": "Não foram identificados eventos nem termos codificados sustentados pelas evidências analisadas.",
                "pontos_atencao": [], "conclusao_preliminar": "Sem indícios com base nas evidências disponíveis."}}

        if not isinstance(relatorio, dict):
            relatorio = {}
        rf = relatorio.get("relatorio_forense")
        if not isinstance(rf, dict):
            rf = relatorio["relatorio_forense"] = {}

        rf["achados"] = [{"evidencia": v["arquivo"], "fonte": v["fonte"], "timestamp": v["timestamp"],
                          "tipo": v["tipo"], "descricao": v["descricao"],
                          "relacionamento_suspeito": v["analise_relacionamento_suspeito"]} for v in eventos]
        _escrever_json(relatorio, "forensic_summary.json")

        texto = _inserir_antes_da_nota(renderizar_markdown(relatorio, consolidado, {"alertas": alertas}),
                                       _secao_grafo(relacoes, eventos) + _secao_geo(geos))
        destino = config.OUTPUT_DIR / "relatorio_forense_final.txt"
        destino.write_text(texto, encoding="utf-8")
        print(f"[Llama] Relatório formatado salvo em: {destino}")
        return relatorio
    finally:
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


if __name__ == "__main__":
    executar_analise_llama()
