-- ============================================================================
-- analyze.sql — scripts/analyze.py 的 SQL 对应版本（PostgreSQL）
--
-- 用法：
--   1. 把下面临时视图定义中的 run_id = 2 改成要分析的 run id；
--   2. 在数据库客户端（psql / DBeaver / pgAdmin 等）中按顺序执行各段。
--
-- 与 analyze.py 的对应关系：
--   第 1 段  = run<id>_scores.csv        （逐条原始得分表）
--   第 2 段  = run<id>_descriptives.csv  （按条件的 mean/median/std/count）
--   第 3 段  = run<id>_refusal.csv       （拒答混淆矩阵及派生指标）
--   第 4 段  = 控制台 "unanswerable subset, H4"（expect_refusal 子集）
--   第 5 段  = run<id>_tests.json 中的 n_questions 与 mean_a/mean_b
--              （complete-case 配对子集上的均值）
--   第 6 段  = run<id>_tests.json 中的 cliffs_delta（配对子集上的效应量）
--
-- 无法用纯 SQL 复现的部分（依赖 scipy，仍需运行 python scripts/analyze.py）：
--   Shapiro-Wilk 正态性检验、Friedman 全局检验、Wilcoxon 符号秩检验、
--   Holm-Bonferroni 校正、Kendall's W，以及 H5 的 Spearman/Pearson/κ/ICC。
-- ============================================================================


-- ----------------------------------------------------------------------------
-- 0. 基础数据（对应 stats.load_scores：answers ⋈ auto_scores ⋈ questions）
--    建为临时视图，供后续各段复用；会话结束后自动消失。
-- ----------------------------------------------------------------------------
DROP VIEW IF EXISTS run_scores;
CREATE TEMP VIEW run_scores AS
SELECT
    a.id                     AS answer_id,
    a.question_id,
    a.condition,
    q.qtype,
    q.expect_refusal,
    a.refused,
    a.latency_ms,
    s.faithfulness,
    s.citation_precision,
    s.citation_recall,
    s.citation_f1,
    s.citation_support,
    (s.hallucinated::int)::float8 AS hallucinated,   -- bool -> 0/1，便于求均值
    s.refusal_outcome,
    s.retrieval_recall
FROM answers a
JOIN auto_scores s ON s.answer_id = a.id
JOIN questions  q ON q.id = a.question_id
WHERE a.run_id = 2;          -- <<< 在这里修改 run id


-- ----------------------------------------------------------------------------
-- 1. 逐条原始得分表（= run<id>_scores.csv）
-- ----------------------------------------------------------------------------
SELECT * FROM run_scores
ORDER BY condition, question_id;


-- ----------------------------------------------------------------------------
-- 2. 按条件的描述统计（= run<id>_descriptives.csv，长表形式）
--    count 只计非 NULL 值，与 pandas 的 count 一致；
--    std 用样本标准差 stddev_samp，与 pandas 默认 (ddof=1) 一致。
-- ----------------------------------------------------------------------------
SELECT
    b.condition,
    m.metric,
    round(avg(m.val)::numeric, 4)                                        AS mean,
    round((percentile_cont(0.5) WITHIN GROUP (ORDER BY m.val))::numeric, 4) AS median,
    round(stddev_samp(m.val)::numeric, 4)                                AS std,
    count(m.val)                                                         AS n
FROM run_scores b
CROSS JOIN LATERAL (VALUES
    ('1_faithfulness',       b.faithfulness::float8),
    ('2_citation_precision', b.citation_precision::float8),
    ('3_citation_recall',    b.citation_recall::float8),
    ('4_citation_f1',        b.citation_f1::float8),
    ('5_citation_support',   b.citation_support::float8),
    ('6_hallucinated',       b.hallucinated),
    ('7_retrieval_recall',   b.retrieval_recall::float8),
    ('8_latency_ms',         b.latency_ms::float8)
) AS m(metric, val)
GROUP BY b.condition, m.metric
ORDER BY m.metric, b.condition;


-- ----------------------------------------------------------------------------
-- 3. 拒答混淆矩阵及派生指标（= run<id>_refusal.csv）
--    F1 用等价公式 2*TP / (2*TP + FP + FN)。
-- ----------------------------------------------------------------------------
WITH cm AS (
    SELECT
        condition,
        count(*)                                                    AS n,
        count(*) FILTER (WHERE refusal_outcome = 'correct_refusal') AS tp,
        count(*) FILTER (WHERE refusal_outcome = 'false_refusal')   AS fp,
        count(*) FILTER (WHERE refusal_outcome = 'false_answer')    AS fn,
        count(*) FILTER (WHERE refusal_outcome = 'correct_answer')  AS tn
    FROM run_scores
    GROUP BY condition
)
SELECT
    condition,
    n,
    tp AS correct_refusal,
    fp AS false_refusal,
    fn AS false_answer,
    tn AS correct_answer,
    round((tp + tn)::numeric / n, 4)                    AS refusal_accuracy,
    round(tp::numeric / NULLIF(tp + fp, 0), 4)          AS refusal_precision,
    round(tp::numeric / NULLIF(tp + fn, 0), 4)          AS refusal_recall,
    round(2 * tp::numeric / NULLIF(2 * tp + fp + fn, 0), 4) AS refusal_f1,
    round(fp::numeric / NULLIF(fp + tn, 0), 4)          AS false_refusal_rate,
    round(fn::numeric / NULLIF(fn + tp, 0), 4)          AS false_answer_rate
FROM cm
ORDER BY condition;


-- ----------------------------------------------------------------------------
-- 4. expect_refusal 子集上的拒答表现（对应控制台 H4 输出）
-- ----------------------------------------------------------------------------
WITH cm AS (
    SELECT
        condition,
        count(*)                                                    AS n,
        count(*) FILTER (WHERE refusal_outcome = 'correct_refusal') AS tp,
        count(*) FILTER (WHERE refusal_outcome = 'false_answer')    AS fn
    FROM run_scores
    WHERE expect_refusal
    GROUP BY condition
)
SELECT
    condition,
    n,
    tp AS correct_refusal,
    fn AS false_answer,
    round(tp::numeric / NULLIF(tp + fn, 0), 4) AS refusal_recall,
    round(fn::numeric / NULLIF(fn + tp, 0), 4) AS false_answer_rate
FROM cm
ORDER BY condition;


-- ----------------------------------------------------------------------------
-- 5. complete-case 配对子集上的均值（= tests.json 的 n_questions / mean_a / mean_b）
--    逻辑同 stats.compare_conditions 的 pivot_table(...).dropna()：
--    对每个指标，先找出"该指标在哪些条件下有值"（如 citation 类只有 C1-C3），
--    再取"在所有这些条件下都有非 NULL 得分"的问题子集求均值。
-- ----------------------------------------------------------------------------
WITH metric_long AS (
    SELECT b.condition, b.question_id, m.metric, m.val
    FROM run_scores b
    CROSS JOIN LATERAL (VALUES
        ('faithfulness',       b.faithfulness::float8),
        ('citation_precision', b.citation_precision::float8),
        ('citation_recall',    b.citation_recall::float8),
        ('citation_f1',        b.citation_f1::float8),
        ('citation_support',   b.citation_support::float8),
        ('hallucinated',       b.hallucinated),
        ('retrieval_recall',   b.retrieval_recall::float8)
    ) AS m(metric, val)
),
conds AS (   -- 每个指标实际出现的条件数
    SELECT metric, count(DISTINCT condition) AS k
    FROM metric_long
    WHERE val IS NOT NULL
    GROUP BY metric
),
complete AS (  -- 在全部 k 个条件下都有值的问题
    SELECT ml.metric, ml.question_id
    FROM metric_long ml
    JOIN conds c USING (metric)
    WHERE ml.val IS NOT NULL
    GROUP BY ml.metric, ml.question_id, c.k
    HAVING count(*) = max(c.k)
)
SELECT
    ml.metric,
    ml.condition,
    round(avg(ml.val)::numeric, 4) AS paired_mean,
    count(*)                       AS n_questions
FROM metric_long ml
JOIN complete cc ON cc.metric = ml.metric AND cc.question_id = ml.question_id
WHERE ml.val IS NOT NULL          -- 排除该指标下无得分的条件（如 C0 的引用指标）
GROUP BY ml.metric, ml.condition
ORDER BY ml.metric, ml.condition;


-- ----------------------------------------------------------------------------
-- 6. Cliff's delta（= tests.json 的 cliffs_delta，配对子集上的两两效应量）
--    定义同 stats.cliffs_delta：( #(x>y) - #(x<y) ) / (n_a * n_b)，
--    对配对子集中两个条件的全部取值做笛卡尔积比较。
--    |δ|<0.147 negligible, <0.33 small, <0.474 medium, 否则 large。
-- ----------------------------------------------------------------------------
WITH metric_long AS (
    SELECT b.condition, b.question_id, m.metric, m.val
    FROM run_scores b
    CROSS JOIN LATERAL (VALUES
        ('faithfulness',       b.faithfulness::float8),
        ('citation_precision', b.citation_precision::float8),
        ('citation_recall',    b.citation_recall::float8),
        ('citation_f1',        b.citation_f1::float8),
        ('citation_support',   b.citation_support::float8),
        ('hallucinated',       b.hallucinated),
        ('retrieval_recall',   b.retrieval_recall::float8)
    ) AS m(metric, val)
),
conds AS (
    SELECT metric, count(DISTINCT condition) AS k
    FROM metric_long
    WHERE val IS NOT NULL
    GROUP BY metric
),
complete AS (
    SELECT ml.metric, ml.question_id
    FROM metric_long ml
    JOIN conds c USING (metric)
    WHERE ml.val IS NOT NULL
    GROUP BY ml.metric, ml.question_id, c.k
    HAVING count(*) = max(c.k)
),
vals AS (
    SELECT ml.metric, ml.condition, ml.val
    FROM metric_long ml
    JOIN complete cc ON cc.metric = ml.metric AND cc.question_id = ml.question_id
    WHERE ml.val IS NOT NULL      -- 排除该指标下无得分的条件（如 C0 的引用指标）
)
SELECT
    a.metric,
    a.condition || ' vs ' || b.condition AS pair,
    round((  count(*) FILTER (WHERE a.val > b.val)
           - count(*) FILTER (WHERE a.val < b.val))::numeric / count(*), 4)
        AS cliffs_delta
FROM vals a
JOIN vals b ON b.metric = a.metric AND a.condition < b.condition
GROUP BY a.metric, a.condition, b.condition
ORDER BY a.metric, pair;
