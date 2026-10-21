"""Dados FICTÍCIOS para testar a GUI e o relatório sem rodar nenhum modelo (--demo)."""
import json
import time
from pathlib import Path

from relatorio import renderizar_markdown


def criar_evidencias_demo(pasta):
    """Cria arquivos vazios + Filelist.csv de exemplo."""
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    for nome in ("a1.ogg", "a2.ogg", "evidence_1.jpg", "evidence_2.jpg", "notas.pdf"):
        (pasta / nome).touch()
    # No formato do IPED (pt-BR): separador ";" e colunas Nome/Ext/Categoria/Caminho
    (pasta / "Lista_de_Arquivos.csv").write_text(
        '"Nome";"Atalho";"Tamanho";"Ext";"Marcador";"Categoria";"Caminho"\n'
        '"a1.ogg";"a1.ogg";"1024";"ogg";"";"Audios";"evidence/a1.ogg"\n'
        '"a2.ogg";"a2.ogg";"2048";"ogg";"";"Audios";"evidence/a2.ogg"\n'
        '"evidence_1.jpg";"evidence_1.jpg";"4096";"jpg";"";"Other Images";"evidence/evidence_1.jpg"\n'
        '"evidence_2.jpg";"evidence_2.jpg";"4096";"jpg";"";"Other Images";"evidence/evidence_2.jpg"\n'
        '"notas.pdf";"notas.pdf";"512";"pdf";"";"PDF Documents";"evidence/notas.pdf"\n'
        '"evidence";"";"4096";"";"";"Folders";"evidence"\n',
        encoding="utf-8")


def gerar_resultados_demo(pasta):
    pasta = Path(pasta)
    visual = [
        {"name": "evidence_1.jpg", "description": "No suspicious elements visible. One person sitting at a table.",
         "visible_text": ""},
        {"name": "evidence_2.jpg", "description": "Two people; one holding a bundle of cash. Several taped packages on a table.",
         "visible_text": "Pix enviado R$ 500,00 - Zé"},
    ]
    audio = [{"name": "a1.ogg", "full_text": "(DEMO) vou distribuir o café nas caixas amanhã, o Zé paga no pix",
              "timestamps": [{"timestamp": "[00:02.10 -> 00:07.80]",
                              "text": "vou distribuir o café nas caixas amanhã, o Zé paga no pix"}]}]
    entidades = {"entities": [
        {"entity_id": "ENT_001", "nome_ou_alias": "Zé", "tipo": "suspeito", "fontes": ["AUDIO", "IMAGEM"],
         "arquivos": ["a1.ogg", "evidence_2.jpg"], "relevancia": "alta"},
        {"entity_id": "ENT_002", "nome_ou_alias": "Pacotes embalados", "tipo": "objeto_de_interesse",
         "fontes": ["IMAGEM"], "arquivos": ["evidence_2.jpg"], "relevancia": "media"}]}
    alertas = {"alertas": [{
        "termo": "café", "significado_provavel": "possível referência a entorpecentes",
        "contexto": "distribuir o café nas caixas amanhã", "arquivo": "a1.ogg",
        "timestamp": "[00:02.10 -> 00:07.80]", "confianca": "media",
        "justificativa": "Termo usado com distribuição e pagamento via pix; imagem mostra pacotes embalados e dinheiro."}]}
    eventos = {"eventos": [{
        "event_id": "EVT_001", "tipo": "entrega", "descricao": "Combinação de distribuição de 'caixas' com pagamento via pix",
        "entidades": ["ENT_001", "ENT_002"], "fonte": "AUDIO", "arquivo": "a1.ogg",
        "timestamp": "[00:02.10 -> 00:07.80]", "evidencias_cruzadas": ["evidence_2.jpg"],
        "interpretacao": "Hipótese: distribuição de entorpecentes"}]}
    relacoes = {"entity_event_relationships": [
        {"entity_id": "ENT_001", "event_id": "EVT_001", "papel": "pagador", "arquivo": "a1.ogg"},
        {"entity_id": "ENT_002", "event_id": "EVT_001", "papel": "objeto", "arquivo": "evidence_2.jpg"}]}
    consolidado = {"entidades": entidades, "eventos": eventos, "relacionamentos": relacoes,
                   "linguagem_codificada": alertas}
    relatorio = {"relatorio_forense": {
        "resumo_executivo": "DADOS FICTÍCIOS DE DEMONSTRAÇÃO. Áudio sugere combinação de entrega com pagamento via pix; "
                            "imagem mostra pacotes e dinheiro, reforçando a hipótese de uso de linguagem codificada.",
        "achados": [{"evidencia": "a1.ogg", "fonte": "AUDIO", "timestamp": "[00:02.10 -> 00:07.80]",
                     "tipo": "entrega", "descricao": "Menção a distribuir 'café' em 'caixas' com pagamento via pix."}],
        "pontos_atencao": ["Termo 'café' em contexto atípico (hipótese)"],
        "conclusao_preliminar": "Indícios compatíveis com distribuição de substâncias; requer validação pericial."}}

    def salvar(nome, dado):
        (pasta / nome).write_text(json.dumps(dado, ensure_ascii=False, indent=4), encoding="utf-8")

    salvar("visual.json", visual)
    salvar("audio.json", audio)
    salvar("entities.json", entidades)
    salvar("linguagem_codificada.json", alertas)
    salvar("events.json", eventos)
    salvar("entity_event_relationships.json", relacoes)
    salvar("consolidado.json", consolidado)
    salvar("forensic_summary.json", relatorio)
    (pasta / "relatorio_forense_final.txt").write_text(
        renderizar_markdown(relatorio, consolidado, alertas), encoding="utf-8")


def executar_demo(pasta, log=print):
    log("[DEMO] Modo de demonstração: nenhum modelo será carregado; os dados abaixo são FICTÍCIOS.")
    for etapa in ("ETAPA 1: IMAGENS (simulado)", "ETAPA 2: ÁUDIOS (simulado)", "ETAPA 3: ANÁLISE FORENSE (simulado)"):
        log(f"=== {etapa} ===")
        time.sleep(0.6)
    gerar_resultados_demo(pasta)
    log("=== PIPELINE CONCLUÍDA (DEMO) ===")
    return True
