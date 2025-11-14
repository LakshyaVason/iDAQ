Run: uvicorn main:app --reload
Then open http://127.0.0.1:8000/docs

## Local Llama agent (offline diagnostics)

`local_llama_agent.py` packages the Jetson-side responsibilities for the new
hybrid setup.  It keeps everything offline by talking to Ollama's
`llama3.2:1b` model that runs directly on the device.

### Responsibilities covered locally

* Check whether the requested Ollama model is already installed and provide a
  helpful hint if it is missing.
* Train (or load) a lightweight RandomForest classifier from `fault.csv`.
* Fit a simple anomaly baseline from `normal.csv` and report outlier sensors.
* Retrieve structured troubleshooting steps from `local_knowledge_base.json`.
* Ask the local LLM to merge retrieval context with real-time sensor snapshots
  to emit a step-by-step diagnostic plan.

### Usage

```bash
python local_llama_agent.py
```

By default the script will:

1. Print the status of the `llama3.2:1b` model on Ollama.
2. Train and persist the RandomForest classifier (artifacts stored under
   `artifacts/`).
3. Fit the anomaly detector and evaluate a demo sensor snapshot.
4. Send the structured-diagnostics prompt to `http://127.0.0.1:11434`. Make
   sure `ollama serve` is running before invoking the script.

```mermaid
flowchart TD
    A["Power Electronics Circuit(Lab Setup)"] --> B["Data Acquisition & Labeling(Oscilloscope / DAQ / Lab)"]
    B --> C["Jetson Nano + CUDA(Edge Compute Node)"]

    %% Models branching from Jetson
    C --> C1["Model 1:Fault Classifier (CNN)"]
    C --> C2["Model 2:Diagnostic LLM (Meta-Llama 3.1 8B)"]
    C --> C3["Model 3:Retrieval Agent (MiniLM / S-BERT)"]
    C --> C4["Model 4:Planner LLM (Gemma 2 / Mistral 7B)"]

    %% Merge models back to coordinator
    C1 --> D["LangChain Coordinator Agent(routes queries & responses)"]
    C2 --> D
    C3 --> D
    C4 --> D

    D --> E["User Interface / API Layer(Web / Terminal App)"]

    %% Styling
    style A fill:#9be7a2,stroke:#2e7d32,stroke-width:2px,color:#000000
    style B fill:#a5d6a7,stroke:#2e7d32,stroke-width:2px,color:#000000
    style C fill:#bbdefb,stroke:#1565c0,stroke-width:2px,color:#000000
    style C1 fill:#90caf9,stroke:#0d47a1,stroke-width:2px,color:#000000
    style C2 fill:#ef9a9a,stroke:#b71c1c,stroke-width:2px,color:#000000
    style C3 fill:#90caf9,stroke:#0d47a1,stroke-width:2px,color:#000000
    style C4 fill:#ef9a9a,stroke:#b71c1c,stroke-width:2px,color:#000000
    style D fill:#ffcc80,stroke:#e65100,stroke-width:2px,color:#000000
    style E fill:#e0e0e0,stroke:#424242,stroke-width:2px,color:#000000
