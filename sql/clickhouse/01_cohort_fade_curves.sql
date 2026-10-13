-- 01 Cohort fade curves: mean capacity normalised to each cell's cycle-10..100 average,
--    per batch and 100-cycle bucket, valid cycles up to end of life only.
WITH ref AS (
    SELECT cell_id, avg(qd) AS q_ref
    FROM cycles
    WHERE cycle >= 10 AND cycle <= 100 AND qd > 0 AND qd <= 1.32
    GROUP BY cell_id
),
v AS (
    SELECT c.cell_id AS cell_id, k.batch AS batch,
           CAST(floor(c.cycle / 100) * 100 AS INTEGER) AS cycle_bucket,
           c.qd / r.q_ref AS q_norm
    FROM cycles c
    JOIN ref r ON c.cell_id = r.cell_id
    JOIN cells k ON c.cell_id = k.cell_id
    WHERE c.qd > 0 AND c.qd <= 1.32 AND c.cycle <= k.cycle_life
)
SELECT batch, cycle_bucket, count(DISTINCT cell_id) AS n_cells,
       round(avg(q_norm), 6) AS mean_q_norm, round(min(q_norm), 6) AS min_q_norm
FROM v
GROUP BY batch, cycle_bucket
ORDER BY batch, cycle_bucket;
