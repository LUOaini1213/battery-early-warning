-- 02 Knee candidates: first cycle >= 101 where the trailing 80-cycle OLS fade rate of discharge
--    capacity exceeds 0.3 mAh/cycle (same window/threshold the Python rolling-slope detector chose
--    on the training cells; SQL uses raw, not median-smoothed, capacity). Only past rows are used.
WITH v AS (
    SELECT cell_id, cycle, qd
    FROM cycles
    WHERE qd > 0 AND qd <= 1.32
),
w AS (
    SELECT cell_id, cycle,
           avg(cycle * qd) OVER win - avg(cycle) OVER win * avg(qd) OVER win AS cov_xy,
           avg(cycle * cycle) OVER win - avg(cycle) OVER win * avg(cycle) OVER win AS var_x,
           count(*) OVER win AS n_win
    FROM v
    WINDOW win AS (PARTITION BY cell_id ORDER BY cycle ROWS BETWEEN 79 PRECEDING AND CURRENT ROW)
)
SELECT w.cell_id AS cell_id, min(w.cycle) AS first_candidate_cycle,
       max(k.cycle_life) AS eol_cycle, max(k.cycle_life) - min(w.cycle) AS lead_to_eol
FROM w
JOIN cells k ON w.cell_id = k.cell_id
WHERE w.cycle >= 101 AND w.n_win = 80 AND -(w.cov_xy / w.var_x) > 0.0003
GROUP BY w.cell_id
ORDER BY cell_id;
