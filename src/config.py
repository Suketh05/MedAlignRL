"""
Central configuration for MedAlign-RL.
Swap MODEL_NAME to scale up on Unity (e.g. Qwen2.5-7B-Instruct) vs. Colab (1.5B).
"""
import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # ---- Paths ----
    project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir: str = field(init=False)
    output_dir: str = field(init=False)
    index_dir: str = field(init=False)

    # ---- Data ----
    dataset_name: str = "har1/MTS_Dialogue-Clinical_Note"  # fallback tried in data.py
    train_split: float = 0.8
    val_split: float = 0.1
    # test_split is the remainder
    seed: int = 42

    # ---- Base model ----
    # Colab-friendly default. Bump to a 7B model on Unity by overriding this.
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_seq_len: int = 2048

    # ---- RAG ----
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_top_k: int = 3

    # ---- Reward model ----
    ner_model_scispacy: str = "en_core_sci_sm"
    ner_model_fallback: str = "en_core_web_sm"
    nli_model_name: str = "cross-encoder/nli-deberta-v3-base"
    reward_weights: dict = field(default_factory=lambda: {
        "entity_f1": 0.4,
        "fact_consistency": 0.4,  # 1 - contradiction_prob from NLI
        "discourse_flow": 0.2,
    })

    # ---- Preference pair generation ----
    n_candidates_per_example: int = 4
    sampling_temperature: float = 0.9
    n_preference_examples: int = 300

    # ---- DPO training ----
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    dpo_beta: float = 0.1
    learning_rate: float = 5e-5
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8

    # ---- Verifier / refiner loop ----
    verifier_score_threshold: float = 0.6
    max_refine_attempts: int = 1

    def __post_init__(self):
        self.data_dir = os.path.join(self.project_root, "data")
        self.output_dir = os.path.join(self.project_root, "outputs")
        self.index_dir = os.path.join(self.data_dir, "rag_index")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)


CFG = Config()
