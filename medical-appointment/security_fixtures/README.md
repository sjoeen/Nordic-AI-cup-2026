# Security fixtures

Versioned adversarial inputs for the inference path. They are synthetic (no
competition audio) and are exercised by `tests/test_security.py` through the
pipeline with a fake ASR backend, and by `tests/test_qa_llm_backend.py` for
the LLM prompt/parse boundary.

`transcript_attacks_v1.json` holds a list of cases:

```json
{
  "id": "direct_override_1",
  "family": "direct_answer_override",
  "units": ["...spoken text as ASR would emit it..."],
  "questions": ["..."],
  "expected": {"schema_valid": true, "not_all_true": true},
  "notes": "what the attack tries to do"
}
```

Passing this suite is evidence for this suite only; it does not prove
immunity to prompt injection.
