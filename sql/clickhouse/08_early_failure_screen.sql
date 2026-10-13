-- 08 Early-failure screen in pure SQL: the 'variance model' (OLS of log10 life on
--    log10 var dQ_100-10(V)) fitted on TRAIN cells, applied to all cells; flag predicted life < 550.
WITH d AS (
    SELECT f.cell_id AS cell_id, f.value AS x, log10(k.cycle_life) AS y,
           k.split AS split, k.cycle_life AS life
    FROM early_features f
    JOIN cells k ON f.cell_id = k.cell_id
    WHERE f.feature = 'log_var_dq'
),
fit AS (
    SELECT (avg(x * y) - avg(x) * avg(y)) / (avg(x * x) - avg(x) * avg(x)) AS slope,
           avg(x) AS mx, avg(y) AS my
    FROM d
    WHERE split = 'train'
)
SELECT d.cell_id AS cell_id, d.split AS split, d.life AS observed_life,
       round(pow(10, fit.my + fit.slope * (d.x - fit.mx)), 3) AS predicted_life,
       CASE WHEN pow(10, fit.my + fit.slope * (d.x - fit.mx)) < 550 THEN 1 ELSE 0 END AS flagged_early_failure,
       CASE WHEN d.life < 550 THEN 1 ELSE 0 END AS actual_early_failure
FROM d
CROSS JOIN fit
ORDER BY cell_id;
