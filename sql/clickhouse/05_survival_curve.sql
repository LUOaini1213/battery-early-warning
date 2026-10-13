-- 05 Survival curve: fraction of cells still above 0.88 Ah at each 100-cycle mark, per batch.
WITH marks AS (SELECT toInt64((number + 1) * 100) AS cycle_mark FROM numbers(24))
SELECT m.cycle_mark AS cycle_mark, k.batch AS batch, count(*) AS n_cells,
       sum(CASE WHEN k.cycle_life > m.cycle_mark THEN 1 ELSE 0 END) AS n_surviving,
       round(sum(CASE WHEN k.cycle_life > m.cycle_mark THEN 1 ELSE 0 END) / count(*), 6) AS frac_surviving
FROM marks m
CROSS JOIN cells k
GROUP BY m.cycle_mark, k.batch
ORDER BY batch, cycle_mark;
