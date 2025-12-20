Qualitative Physics Reasoning (Self-Generated) Dataset
====================================================

Files:
- physics_qualitative_train.jsonl: Train split (JSONL)
- physics_qualitative_test.jsonl:  Test split  (JSONL)

Each JSON object has fields:
- split: "train" or "test"
- domain: "newton" | "friction" | "inertia"
- target_quantity: e.g. "acceleration", "velocity", "friction force"
- intervention: e.g. "mass_up", "force_down", "roughness_up"
- fixed_condition: e.g. "force_constant", "mass_constant", "none"
- prompt: the natural-language question
- label: "increase" | "decrease" | "remain constant"
- is_counterfactual: bool
    - False: physically correct label given the described conditions (ground truth)
    - True:  label intentionally flipped to one of the other two options, while keeping the prompt unchanged

Counts:
- Train samples: 990
- Test  samples: 495
- Total samples: 1485

Notes:
- Counterfactuals are created by changing ONLY the label, leaving the prompt unchanged.
- Prompts are paraphrased with multiple templates and small distractors to reduce template memorization.
