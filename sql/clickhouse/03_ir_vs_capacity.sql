-- 03 Internal resistance vs capacity per cell (valid cycles 2..EOL): correlation and IR growth.
WITH v AS (
    SELECT c.cell_id AS cell_id, c.cycle AS cycle, c.qd AS qd, c.ir AS ir
    FROM cycles c
    JOIN cells k ON c.cell_id = k.cell_id
    WHERE c.qd > 0 AND c.qd <= 1.32 AND c.ir > 0 AND c.cycle >= 2 AND c.cycle <= k.cycle_life
),
m AS (
    SELECT cell_id, min(cycle) AS first_cycle, max(cycle) AS last_cycle FROM v GROUP BY cell_id
)
SELECT v.cell_id AS cell_id, count(*) AS n_cycles,
       round(corr(v.ir, v.qd), 6) AS corr_ir_qd,
       round(1000 * avg(CASE WHEN v.cycle < m.first_cycle + 20 THEN v.ir END), 6) AS ir_start_mohm,
       round(1000 * avg(CASE WHEN v.cycle > m.last_cycle - 20 THEN v.ir END), 6) AS ir_end_mohm,
       round(1000 * (avg(CASE WHEN v.cycle > m.last_cycle - 20 THEN v.ir END)
                   - avg(CASE WHEN v.cycle < m.first_cycle + 20 THEN v.ir END)), 6) AS ir_growth_mohm
FROM v
JOIN m ON v.cell_id = m.cell_id
GROUP BY v.cell_id
ORDER BY cell_id;
