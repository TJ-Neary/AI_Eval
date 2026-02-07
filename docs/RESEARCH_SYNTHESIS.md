# AI_Eval Research Synthesis

> **Purpose**: Synthesized findings from multi-source research on LLM evaluation methodology, benchmarking best practices, hardware profiling, and scoring rigor. These findings directly informed technical decisions TD-010 through TD-013 in the [DevPlan](../DevPlan.md).
>
> **Sources**: Independent research sessions conducted with Claude (Anthropic), Gemini (Google), and ChatGPT (OpenAI) — each tasked with reviewing the AI_Eval architecture and identifying gaps against current industry practices.
>
> **Date**: 2026-02-02

---

## Executive Summary

All three research sources converge on several critical themes that were missing or underdeveloped in the initial AI_Eval design. The most significant findings relate to statistical rigor requirements (naive mean/std dev is insufficient for reliable model comparison), benchmark contamination risks (11-22% performance inflation on widely-seen datasets), and LLM-as-Judge bias patterns (position, verbosity, and self-evaluation biases). These findings have been incorporated into the development plan as technical decisions TD-010 through TD-013.

---

## Gaps Identified in Initial Design

### Critical Priority

| Gap | Initial State | Research Finding |
|-----|---------------|------------------|
| **Statistical Rigor** | Basic mean ± std dev | Need 800+ samples for 3% effect detection, clustered SEs, confidence intervals |
| **Benchmark Contamination** | Not addressed | 11-22% performance inflation on contaminated data; need ITD decontamination |
| **LLM-as-Judge Bias** | Not addressed | Position bias, verbosity bias, self-evaluation bias all documented |
| **Drift Detection** | Not addressed | 75% of LLM deployments show degradation without monitoring |
| **TCO Break-Even Calculator** | Mentioned but not specified | Clear thresholds: <10K/mo favors API, >100K/mo favors local |
| **Memory Bandwidth Metrics** | Not tracked | Key bottleneck for local inference; more important than compute |
| **Thermal Throttle Testing** | Not addressed | Consumer GPUs lose 30% speed after 5+ min; soak testing required |

### Important Priority

| Gap | Research Finding |
|-----|------------------|
| **RAG Evaluation (RAGAS)** | Faithfulness, Answer Relevancy, Context Precision, Context Recall |
| **Calibration Testing** | Does model confidence correlate with correctness? |
| **Quantization Comparison** | Q5_K_M is optimal (65% size reduction, 97-98% quality retention) |
| **Needle-in-Haystack** | Test actual vs. claimed context window capacity |
| **Multi-Language Testing** | 30%+ performance gap between English and low-resource languages |
| **Cold Start vs Hot Start** | Loading 70B model takes 8-13 seconds; must measure separately |
| **Carbon/Energy Tracking** | Up to 70x difference in energy per query between models |

---

## Key Methodologies to Adopt

### 1. Statistical Rigor (From Claude Research)

**Anthropic's 5 Core Practices**:
1. Use CLT for standard errors with 95% CI
2. Cluster SEs for non-independent questions (naive SEM underestimates 3x)
3. Reduce within-question variance via resampling
4. Use paired-difference analysis
5. Conduct power analysis (870 questions for 3% effect at 80% power)

**AI_Eval Action**: Increase minimum test count to 100+ per category; implement bootstrap CIs.

### 2. LLM-as-Judge Best Practices (All Sources)

| Bias | Mitigation |
|------|------------|
| Position Bias | Randomize answer order in comparisons |
| Verbosity Bias | Penalize unnecessary length in rubric |
| Self-Evaluation Bias | Never have model judge its own output |
| Cost | Use small fine-tuned judges (Galileo Luna: $0.02/M tokens) |

**AI_Eval Action**: Add judge configuration with bias mitigation; support local judge models.

### 3. Hardware Profiling (From Gemini Research)

**Memory Bandwidth Formula**:
```
Max TPS = Memory_Bandwidth_GB/s / Model_Size_GB
e.g., RTX 4090 (1 TB/s) with 70B (140GB) = ~7 TPS theoretical max
```

**Soak Test Protocol**:
- Run continuous inference 15-30 min
- Log GPU temp and clock every 30s
- Calculate: `Degradation% = (TPS_initial - TPS_final) / TPS_initial`

**AI_Eval Action**: Add memory bandwidth utilization (MBU) metric; implement soak test mode.

### 4. TCO Break-Even Thresholds (From Claude Research)

| Monthly Queries | Recommendation | Break-Even |
|-----------------|----------------|------------|
| <10K | Cloud API | N/A |
| 10K-100K | Careful evaluation | 12-36 months |
| >100K | Likely local | 6-18 months |
| >1M | Strongly local | 3-12 months |

**AI_Eval Action**: Implement `ai-eval estimate-tco` command with break-even calculator.

### 5. Benchmark Saturation & Contamination (All Sources)

**Saturated Benchmarks to De-Emphasize**:
- MMLU (saturated Sep 2024, 9%+ error rate in questions)
- HumanEval (164 problems widely contaminated)

**Contamination-Resistant Alternatives**:
- LiveCodeBench (weekly new problems)
- SWE-Bench Verified (500 human-validated)
- GPQA Diamond (PhD-level science)

**Detection Method**: Check for 11-22% score difference between known-contaminated and fresh versions.

### 6. Continuous Drift Monitoring (From Claude Research)

**Key Finding**: 75% of businesses see AI performance decline without monitoring. Models unchanged for 6+ months show 35% error rate increase.

**Monitor**:
- Response length variance (GPT-4 showed 23% over time)
- Instruction adherence
- User feedback thumbs down rate
- Population Stability Index (PSI)

**AI_Eval Action**: Add `ai-eval monitor` command for production tracking.

---

## Features to Add (Priority Order)

### Phase 1 Additions (Now)

1. **Bootstrap Confidence Intervals** - On all metrics
2. **Memory Bandwidth Tracking** - Via `nvidia-smi` or `ioreg`
3. **Judge Bias Mitigation** - Randomize order, penalize length

### Phase 2 Additions (Foundation)

4. **Soak Test Mode** - 30-min thermal stability test
5. **Cold Start Timing** - Separate model load time from inference
6. **Contamination Detection** - Compare vs fresh test variants

### Phase 3 Additions (Advanced)

7. **RAGAS Integration** - For RAG pipeline evaluation
8. **Drift Monitor** - Production performance tracking
9. **TCO Calculator** - Break-even analysis tool
10. **Energy Tracking** - Joules per token, CO2 per query

---

## Industry Standard Tools to Integrate

| Tool | Purpose | Integration Priority |
|------|---------|---------------------|
| **lm-evaluation-harness** | Standard benchmarks | High - use as optional backend |
| **RAGAS / DeepEval** | RAG evaluation | **High** - core use case |
| **MTEB** | Embedding evaluation | **High** - for embedding models |
| **Garak** | Red-teaming / jailbreaks | Medium - for safety tests |
| **OpenTelemetry** | Observability standard | Phase 3 - enterprise |

---

## Prompt Engineering Research

> Based on 2024-2025 research on model-specific prompting differences.

### Universal Best Practices

- **Clarity & Specificity**: Ambiguity is primary cause of poor output
- **Structured Prompts**: System message + instructions + context + examples + constraints
- **Few-Shot Learning**: Provide 3-5 examples for pattern matching
- **Chain-of-Thought (CoT)**: "Think step by step" improves reasoning
- **Output Constraints**: Explicitly state format, length, structure

### Model-Specific Differences

| Model Family | Prompting Style | Key Techniques |
|--------------|----------------|----------------|
| **GPT (OpenAI)** | Streamlined single prompt | Crisp numeric constraints, JSON hints, 128K context |
| **Claude (Anthropic)** | XML-tagged sections | `<instructions>`, `<thinking>` tags; 200K context; direct instructions |
| **Llama (Meta)** | Structure-sensitive | System instructions critical; template-based; sensitive to prompt quality |
| **Gemini (Google)** | Flexible, multimodal | Handles mixed media; benefits from role assignment |

### AI_Eval Prompt Engineering Features

1. **Prompt Sensitivity Testing** (Category 7)
   - Same question, 5 different phrasings → measure variance
   - Model robustness score

2. **Prompt Optimization Recommendations**
   - After test run, suggest model-specific prompt improvements
   - E.g., "Claude performs better with XML tags for this task"

3. **Prompt Library** (Documentation)
   - Best prompt templates per model family
   - Domain-specific prompt patterns (RAG, code, analysis)

---

## Commercial Landscape Insights (From Claude Research)

**Market Gap AI_Eval Can Fill**:
- **Local-first evaluation** (most tools are cloud-first)
- **Business metric alignment** (tools focus on technical metrics)
- **Domain-specific evaluators** (healthcare, legal, finance underserved)
- **Non-technical access** (most require engineering expertise)

---

## Summary: DevPlan Updates Required

### New Technical Decisions

- **TD-010**: Statistical Rigor Requirements (min samples, bootstrap CIs)
- **TD-011**: LLM-as-Judge Bias Mitigation Protocol
- **TD-012**: Thermal & Hardware Profiling Standards
- **TD-013**: Contamination Detection Strategy

### New Features to Add to Phase 1-2

- Memory bandwidth utilization metric
- Bootstrap confidence intervals on all scores
- Separate cold start vs inference timing
- Soak test mode (30-min continuous)

### New Features for Phase 3+

- RAGAS integration for RAG use cases
- Production drift monitoring
- TCO break-even calculator
- Energy/carbon tracking

---

## Items for Future Phases

The following items from the research are valuable but **deferred to later phases** based on priority:

### Phase 3-4: Cross-Platform & Consulting Readiness

| Item | Source | Target Phase | Notes |
|------|--------|--------------|-------|
| **Windows Support** | User | Phase 2-3 | Family machines for practice, then client hardware |
| **Multi-machine Result Aggregation** | Claude | Phase 4 | Run on client hardware, aggregate centrally |
| **White-label/Branded Reports** | User | Phase 4 | Client-facing deliverables |
| **vLLM/SGLang Production Servers** | Claude | Phase 4 | High-throughput for enterprise workloads |
| **OpenTelemetry Full Integration** | All | Phase 3 | Enterprise observability standard |

### Phase 4-5: Enterprise & Compliance

| Item | Source | Target Phase | Notes |
|------|--------|--------------|-------|
| **SOC 2 Documentation** | Claude | Phase 5 | Clients may require compliance attestation |
| **EU AI Act 10-year Retention** | Claude | Phase 5 | Required if commercializing in EU |
| **HIPAA Considerations** | Claude | Phase 5 | Healthcare client requirements |
| **Router/Drafter Model Evaluation** | Gemini | Phase 5 | Hybrid routing architecture testing |

### Phase 5+: Advanced Features

| Item | Source | Target Phase | Notes |
|------|--------|--------------|-------|
| **LongBench v2 (2M token tests)** | Claude | Phase 5+ | When frontier models become local-viable |
| **BFCL 1,680 Python Scenarios** | Claude | Phase 5+ | Comprehensive function calling suite |
| **Region-based Carbon Intensity** | Gemini | Phase 5+ | Advanced energy tracking |

---

## Items Out of Scope (Not Applicable)

| Item | Source | Reason |
|------|--------|--------|
| **Multi-tenant SaaS Architecture** | Claude/ChatGPT | Tool is CLI-first, not hosted service |
| **Crowdsourced Evaluation** | Claude/ChatGPT | No crowd infrastructure; use LLM-as-Judge |
| **Inter-rater Reliability (Cohen's κ)** | Claude/ChatGPT | Single-user primary context |
| **Model Fine-Tuning Integration** | All | Evaluation tool, not training tool |
| **MLPerf Formal Compliance** | Claude/ChatGPT | Industry comparison not required |
| **CROWDLAB Algorithm** | Claude | Requires annotator pool |

---

## Integration Status

The following items from this research have been incorporated into the development plan:

| Finding | Action Taken | Reference |
|---------|-------------|-----------|
| Statistical rigor requirements | Added as TD-010 | [DevPlan.md — TD-010](../DevPlan.md#td-010-statistical-rigor-requirements) |
| LLM-as-Judge bias mitigation | Added as TD-011, implemented in `src/scoring/llm_judge.py` | [DevPlan.md — TD-011](../DevPlan.md#td-011-llm-as-judge-bias-mitigation) |
| Hardware profiling & thermal testing | Added as TD-012 | [DevPlan.md — TD-012](../DevPlan.md#td-012-hardware-profiling--thermal-testing) |
| Benchmark contamination strategy | Added as TD-013 | [DevPlan.md — TD-013](../DevPlan.md#td-013-benchmark-contamination-strategy) |
| RAG evaluation (RAGAS) | Elevated to high priority, implemented via DeepEval in `src/scoring/rag_metrics.py` | Phase 2.5 in DevPlan |
| Memory bandwidth utilization | Added to Phase 2 metrics | TD-012 |

### Pending Integration

- Prompt engineering sensitivity testing (Category 7 proposal)
- Additional metrics in TEST_SUITE_SPEC.md (calibration, drift monitoring)
- TCO break-even calculator CLI command
