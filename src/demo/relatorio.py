"""Renderiza o relatório forense final em Markdown/texto (sem dependências pesadas)."""
from datetime import datetime


def _lista(obj, *chaves):
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for c in chaves:
            v = obj.get(c)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
    return []


def _txt(v):
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return ", ".join(str(x) for x in v)
    return str(v)


def renderizar_markdown(relatorio, consolidado, alertas):
    r = relatorio.get("relatorio_forense", relatorio) if isinstance(relatorio, dict) else {}
    if not isinstance(r, dict):
        r = {}
    ents = _lista(consolidado.get("entidades"), "entities", "entidades")
    evts = _lista(consolidado.get("eventos"), "eventos", "events")
    rels = _lista(consolidado.get("relacionamentos"), "entity_event_relationships", "relacionamentos")
    alertas = _lista(alertas, "alertas", "linguagem_codificada")

    nome_ent = {str(e.get("entity_id")): e.get("nome_ou_alias", "?") for e in ents}
    info_evt = {str(e.get("event_id")): e for e in evts}

    L = ["# RELATÓRIO FORENSE", f"Gerado em: {datetime.now():%d/%m/%Y %H:%M}", ""]

    # 1. Resumo executivo
    L += ["## 1. Resumo Executivo", ""]
    if "raw_output" in r:
        L += ["(O modelo não devolveu JSON válido; saída bruta abaixo)", str(r["raw_output"])]
    else:
        L.append(_txt(r.get("resumo_executivo") or r.get("resumo_geral")) or "Sem resumo gerado.")
    L.append("")

    # 2. Entidades
    L += ["## 2. Mapeamento de Entidades", ""]
    if not ents:
        L.append("Nenhuma entidade relevante identificada.")
    por_tipo = {}
    for e in ents:
        por_tipo.setdefault(_txt(e.get("tipo")).lower() or "outros", []).append(e)
    for tipo in sorted(por_tipo):
        L.append(f"### {tipo.replace('_', ' ').capitalize()}")
        for e in por_tipo[tipo]:
            partes = [f"fontes: {_txt(e.get('fontes') or e.get('fonte'))}",
                      f"arquivos: {_txt(e.get('arquivos') or e.get('arquivo'))}"]
            if e.get("relevancia"):
                partes.append(f"relevância: {_txt(e.get('relevancia'))}")
            L.append(f"- **{_txt(e.get('nome_ou_alias'))}** ({_txt(e.get('entity_id'))}) — " + " | ".join(partes))
        L.append("")

    # 3. Redes de relacionamento
    L += ["## 3. Redes de Relacionamento", ""]
    por_evento = {}
    for rel in rels:
        por_evento.setdefault(str(rel.get("event_id")), []).append(rel)
    if not por_evento:
        L.append("Nenhuma relação entre entidades e eventos identificada.")
    for eid, lista in por_evento.items():
        ev = info_evt.get(eid, {})
        cab = f"**{eid}** [{_txt(ev.get('tipo'))}] {_txt(ev.get('descricao'))}"
        local = " ".join(x for x in (_txt(ev.get("arquivo")), _txt(ev.get("timestamp"))) if x)
        L.append(f"- {cab}" + (f" ({local})" if local else ""))
        for rel in lista:
            nome = nome_ent.get(str(rel.get("entity_id")), _txt(rel.get("entity_id")))
            L.append(f"    - {nome} → {_txt(rel.get('papel')) or 'participante'}")
    L.append("")

    # 4. Linguagem codificada
    L += ["## 4. Alertas de Linguagem Codificada", ""]
    if not alertas:
        L.append("Nenhum termo com indício de linguagem codificada.")
    for a in alertas:
        L.append(f"- [!] \"{_txt(a.get('termo'))}\" → hipótese: {_txt(a.get('significado_provavel'))} "
                 f"(confiança: {_txt(a.get('confianca'))})")
        if a.get("contexto"):
            L.append(f"    - Contexto: {_txt(a.get('contexto'))}")
        local = " ".join(x for x in (_txt(a.get("arquivo")), _txt(a.get("timestamp"))) if x)
        if local:
            L.append(f"    - Evidência: {local}")
        if a.get("justificativa"):
            L.append(f"    - Justificativa: {_txt(a.get('justificativa'))}")
    L.append("")

    # 5. Achados
    achados = _lista(r.get("achados"), "achados")
    if achados:
        L += ["## 5. Achados por Evidência", ""]
        for i, a in enumerate(achados, 1):
            L.append(f"{i}. [{_txt(a.get('fonte')) or '?'}] {_txt(a.get('evidencia')) or '?'} | "
                     f"{_txt(a.get('timestamp')) or 'sem timestamp'} | {_txt(a.get('tipo'))}")
            L.append(f"   {_txt(a.get('descricao'))}")
        L.append("")

    pontos = r.get("pontos_atencao")
    if pontos:
        L += ["## Pontos de Atenção", ""] + [f"- {_txt(p)}" for p in (pontos if isinstance(pontos, list) else [pontos])] + [""]
    if r.get("conclusao_preliminar"):
        L += ["## Conclusão Preliminar", "", _txt(r.get("conclusao_preliminar")), ""]

    L += ["---",
          "Nota: interpretações de linguagem codificada e cruzamentos são HIPÓTESES geradas automaticamente "
          "e exigem validação pericial antes de qualquer conclusão."]
    return "\n".join(L)
