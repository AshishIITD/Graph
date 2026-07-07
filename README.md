# Graph Ecosystem Masterclass (2026+)

A production-grade, from‑scratch reference and teaching codebase for graph engineering: classical graph algorithms, spectral theory, node and knowledge‑graph embeddings, Graph Neural Networks ([...])

---

## Highlights

- Six progressive learning phases implemented as runnable Python modules (foundations → algorithms → embeddings/PGMs → knowledge graphs/KGEs → GNNs → GraphRAG & agents).
- Live, client-side glassmorphic dashboard for algorithm visualization (dashboard/index.html).
- Pre-built publication PDF (Graph_Ecosystem_Masterclass.pdf) plus a script to regenerate it.
- Minimal external dependencies; designed to be runnable on CPU for demos.

---

## Quick start (recommended)

Requirements: Python 3.8+ and a working pip. CPU-only PyTorch works for the included demos.

1. Clone and install Python deps

```bash
git clone https://github.com/AshishIITD/Graph.git
cd Graph
pip install -r requirements.txt
```

2. (Optional) Install PDF build dependencies

The textbook generator (`generate_masterclass_pdf.py`) prints `masterclass_textbook.html` to PDF using a headless browser. Two common ways to provide a headless Chromium:

- Playwright (recommended, cross-platform):

```bash
pip install playwright
python -m playwright install chromium
```

- System Chromium / Chrome + chromedriver or other tooling: ensure `chromium`/`google-chrome` is on PATH and accessible from Python tooling.

3. Run the demo verification suite

```bash
python3 run_demo.py
```

This runs small, self‑contained demonstrations and sanity checks for every phase. Expect printed progress and short numerical checks; the demos are designed to run on CPU.

4. Generate the textbook PDF

```bash
python3 generate_masterclass_pdf.py
```

If you installed Playwright, the script will use the installed headless Chromium. If you rely on a system Chrome binary, ensure it's available and that the script is configured to call it.

5. Open the interactive sandbox

Open `dashboard/index.html` in any modern browser to interact with force-directed visualizations and live algorithm animations.

---

## Project layout

```text
Graph/
├── dashboard/                 # Glassmorphic front-end sandbox (static HTML/JS/CSS)
├── src/                       # Masterclass Python package with six phase modules
│   ├── __init__.py
│   ├── phase1_foundations.py
│   ├── phase2_algorithms.py
│   ├── phase3_embeddings_pgm.py
│   ├── phase4_knowledge_graphs.py
│   ├── phase5_gnns.py
│   └── phase6_graphrag_agents.py
├── run_demo.py                # Automated demo + verification suite
├── generate_masterclass_pdf.py# Render HTML -> PDF (requires headless Chromium)
├── Graph_Ecosystem_Masterclass.pdf # Pre-built textbook (binaries may be large)
├── requirements.txt           # Minimal Python dependencies
└── .gitignore
```

---

## Notes, recommendations and troubleshooting

- requirements.txt contains core Python libraries (torch, numpy, scipy, networkx). For reproducible environments consider pinning exact versions or adding a pyproject.toml / environment.yml.
- PDF generation requires a headless browser. Playwright is cross-platform and simple to install; the README above provides the commands.
- The repository currently contains a pre-built PDF. If you want smaller repo size, consider storing generated assets in releases rather than in-tree.
- run_demo.py is CPU-friendly but uses PyTorch; if you have a GPU and want faster training, ensure an appropriate CUDA-enabled PyTorch wheel is installed before `pip install -r requirements.txt`.
- If generate_masterclass_pdf.py fails with browser errors, install Playwright and run `python -m playwright install chromium`, or check that your system Chromium/Chrome binary is reachable.

---

## Suggested CI (example)

A minimal CI job should:

- Use Python 3.8+ runner
- pip install -r requirements.txt
- Run `python3 run_demo.py` on CPU (no GPU required for the demo)
- (Optional) Install Playwright and run the PDF builder to verify build steps

You can add a lightweight GitHub Actions workflow at `.github/workflows/ci.yml` that performs the steps above.

---

## Contributing

Contributions, bug reports, and improvements are welcome. If you plan to submit code:

- Keep changes small and focused.
- Add unit tests or update `run_demo.py` to include regression checks for new functionality.
- If adding heavy generated assets (PDFs, models), prefer attaching them to a release instead of committing large binaries to main branches.

---
