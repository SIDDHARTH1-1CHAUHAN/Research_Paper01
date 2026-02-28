# Agent Architecture

| Agent / Phase | Primary Model Used | Key Tasks Performed |
|---|---|---|
| Input Ingestion Agent | GPT-4o-mini (Primary) | Performs FOL-guided claim decomposition and filters out non-verifiable subclaims. |
| Query Generation Agent | GPT-4o-mini (Primary) | Transforms atomic subclaims into multiple SEO-optimized search queries. |
| Evidence Seeking Agent | Gemini 1.5 Flash | Processes full-text web content to isolate relevant passages from credible sources. |
| Verdict Prediction Agent | GPT-4o-mini (Primary) | Synthesizes evidence using a voting mechanism to produce final labels and explanations. |
| Evaluation (LLM-as-Judge) | GPT-4 | Ranks the quality of generated explanations based on Coverage, Soundness, and Readability. |

## Evaluation with HoVER and FEVEROUS Datasets

When evaluating the system's performance using the F1 score on the **HoVER** and **FEVEROUS** datasets, please adhere to the following constraint:

- **Exclude Supporting Claims**: Do not use supporting claims from the HoVER and FEVEROUS datasets during this evaluation. The F1 score, representing the harmonic mean of precision and recall, should be calculated solely on the model's ability to identify and process non-supported or refuted claims within these specific datasets. This ensures a rigorous assessment of the system's claim verification and refutation capabilities.
