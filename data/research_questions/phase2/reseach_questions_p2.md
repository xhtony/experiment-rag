RQ1: How do retrieval and answer-control strategies affect the reliability of question answering over university course and policy documents?
- RQ1a: To what extent do standard vector RAG, hybrid retrieval with reranking, and corrective/refusal-aware RAG improve answer faithfulness compared with a no-RAG baseline?
- RQ1b: Does hybrid retrieval with reranking improve citation precision and citation completeness compared with standard vector RAG?
- RQ1c: Does corrective/refusal-aware RAG reduce claim-level hallucination compared with standard vector RAG?
- RQ1d: How does corrective/refusal-aware RAG affect appropriate refusal and clarification behaviour on unanswerable and ambiguous questions?

RQ2: How valid are automated RAG evaluation metrics, and what failure patterns do they reveal in university course and policy question answering?
- RQ2a: How closely do automated measures of faithfulness, citation accuracy, hallucination, and refusal correctness align with manual rubric-based evaluation?
- RQ2b: How does automated–manual evaluation alignment vary across question types, system conditions, and error categories?
- RQ2c: Where do the principal failures occur within the RAG pipeline, including retrieval, ranking, evidence use, citation attribution, generation, and answer-versus-abstain decisions?

RQ3: Can a hybrid corrective and refusal-aware RAG strategy improve the balance between evidence-grounded answering and appropriate abstention?
- RQ3a: Does the proposed hybrid corrective and refusal-aware condition, C4, improve retrieval recall and citation completeness compared with corrective/refusal-aware RAG using dense retrieval, C3?
- RQ3b: Does C4 reduce false refusals compared with C3 on answerable multi-document and conditional questions?
- RQ3c: Does C4 reduce false answers compared with hybrid retrieval and reranking alone, C2, on unanswerable and ambiguous questions?
- RQ3d: Does C4 achieve a better overall balance among faithfulness, citation grounding, hallucination, appropriate refusal, answer coverage, and latency than C2 and C3?
