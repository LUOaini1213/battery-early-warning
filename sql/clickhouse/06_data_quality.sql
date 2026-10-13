-- 06 Data quality per cell: placeholder / implausible values and temperature range.
SELECT cell_id, count(*) AS n_rows,
       sum(CASE WHEN qd <= 0 THEN 1 ELSE 0 END) AS n_qd_nonpositive,
       sum(CASE WHEN qd > 1.32 THEN 1 ELSE 0 END) AS n_qd_above_physical,
       sum(CASE WHEN ir <= 0 THEN 1 ELSE 0 END) AS n_ir_nonpositive,
       round(min(CASE WHEN tavg > 0 THEN tavg END), 4) AS tavg_min_c,
       round(max(tmax), 4) AS tmax_max_c
FROM cycles
GROUP BY cell_id
ORDER BY cell_id;
