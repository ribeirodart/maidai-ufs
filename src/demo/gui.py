"""Interface gráfica (CustomTkinter)."""
import json
import os
import queue
import sys
import threading
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import pipeline  # noqa: E402

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class App(ctk.CTk):
    def __init__(self, evidence_dir=None, filelist=None, searchapp=None):
        super().__init__()
        self.title("Pipeline Forense Multimodal")
        self.geometry("1100x760")

        self.evidence_dir = str(evidence_dir or config.EVIDENCE_DIR)
        self.filelist = str(filelist or config.achar_filelist(self.evidence_dir))
        self.csv_manual = filelist is not None
        self.searchapp = str(searchapp or "")
        self.fila = queue.Queue()

        topo = ctk.CTkFrame(self)
        topo.pack(fill="x", padx=10, pady=10)
        topo.columnconfigure(1, weight=1)
        self.lbl_csv = self._linha(topo, 0, "CSV do IPED (lista de arquivos)", self.escolher_csv, self.filelist)
        self.lbl_exe = self._linha(topo, 1, "IPED-SearchApp.exe", self.escolher_exe, self.searchapp or "(não selecionado)")
        self.lbl_dir = self._linha(topo, 2, "Pasta das evidências", self.escolher_pasta, self.evidence_dir)

        botoes = ctk.CTkFrame(topo, fg_color="transparent")
        botoes.grid(row=3, column=0, columnspan=2, pady=(8, 4))
        self.btn_run = ctk.CTkButton(botoes, text="▶ Executar pipeline", command=self.iniciar)
        self.btn_run.pack(side="left", padx=6)
        ctk.CTkButton(botoes, text="🔎 Abrir SearchApp", command=self.abrir).pack(side="left", padx=6)

        self.abas = ctk.CTkTabview(self)
        self.abas.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.txt = {}
        for nome in ("Log", "Relatório", "Entidades", "Eventos", "Imagens", "Áudios"):
            self.abas.add(nome)
            box = ctk.CTkTextbox(self.abas.tab(nome), wrap="word")
            box.pack(fill="both", expand=True)
            self.txt[nome] = box

        self.protocol("WM_DELETE_WINDOW", self.fechar)
        self.after(100, self._poll)

    def _linha(self, pai, linha, texto, cmd, valor):
        ctk.CTkButton(pai, text=texto, width=200, command=cmd).grid(row=linha, column=0, padx=6, pady=3, sticky="w")
        lbl = ctk.CTkLabel(pai, text=valor, anchor="w")
        lbl.grid(row=linha, column=1, padx=6, sticky="ew")
        return lbl

    # --- seleção de arquivos ---
    def escolher_csv(self):
        p = filedialog.askopenfilename(title="Selecione o CSV exportado pelo IPED",
                                       filetypes=[("CSV", "*.csv"), ("Todos", "*.*")])
        if p:
            self.filelist, self.csv_manual = p, True
            self.lbl_csv.configure(text=p)

    def escolher_exe(self):
        p = filedialog.askopenfilename(title="Selecione o IPED-SearchApp.exe (na pasta do caso IPED)",
                                       filetypes=[("Executável", "*.exe"), ("Todos", "*.*")])
        if p:
            self.searchapp = p
            self.lbl_exe.configure(text=p)

    def escolher_pasta(self):
        p = filedialog.askdirectory(title="Pasta dos arquivos listados no CSV")
        if p:
            self.evidence_dir = p
            self.lbl_dir.configure(text=p)
            if not self.csv_manual:
                self.filelist = str(config.achar_filelist(p))
                self.lbl_csv.configure(text=self.filelist)

    # --- execução ---
    def iniciar(self):
        if not os.path.isdir(self.evidence_dir):
            messagebox.showerror("Erro", "Selecione uma pasta de evidências válida.")
            return
        self.btn_run.configure(state="disabled")
        self.txt["Log"].delete("1.0", "end")
        self.abas.set("Log")
        threading.Thread(target=self._trabalho, daemon=True).start()

    def _trabalho(self):
        try:
            ok = pipeline.executar_pipeline(self.evidence_dir, self.filelist,
                                            log=lambda s: self.fila.put(("log", s + "\n")))
            self.fila.put(("done", ok))
        except Exception as e:
            self.fila.put(("erro", str(e)))

    def _poll(self):
        try:
            while True:
                tipo, dado = self.fila.get_nowait()
                if tipo == "log":
                    self.txt["Log"].insert("end", dado)
                    self.txt["Log"].see("end")
                elif tipo == "erro":
                    self.txt["Log"].insert("end", f"\n[ERRO] {dado}\n")
                    self.btn_run.configure(state="normal")
                elif tipo == "done":
                    self.btn_run.configure(state="normal")
                    self.mostrar()
                    if self.searchapp:
                        self.abrir(silencioso=True)
        except queue.Empty:
            pass
        self.after(100, self._poll)

    # --- evidências na tela ---
    def _set(self, aba, texto):
        self.txt[aba].delete("1.0", "end")
        self.txt[aba].insert("end", texto)

    def mostrar(self):
        d = pipeline.pasta_saida(self.evidence_dir)
        self._set("Relatório", pipeline.formatar_relatorio(d))
        self._set("Entidades", json.dumps(pipeline.ler_json(d, "entities.json"), ensure_ascii=False, indent=2))
        self._set("Eventos", json.dumps(pipeline.ler_json(d, "events.json"), ensure_ascii=False, indent=2))
        self._set("Imagens", pipeline.texto_imagens(d))
        self._set("Áudios", pipeline.texto_audios(d))
        self.abas.set("Relatório")

    def abrir(self, silencioso=False):
        ok, msg = pipeline.abrir_searchapp(self.searchapp)
        if not ok and not silencioso:
            messagebox.showwarning("SearchApp", msg)

    def fechar(self):
        pipeline.encerrar_processos()
        self.destroy()
