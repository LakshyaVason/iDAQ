Run: uvicorn main:app --reload
Then open http://127.0.0.1:8000/docs

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
