-- 04 Early-life feature ranking: Pearson correlation of each first-100-cycle feature with
--    log10(cycle life), TRAIN cells only (ranking on test cells would leak).
SELECT f.feature AS feature, count(*) AS n_train_cells,
       round(corr(f.value, log10(k.cycle_life)), 6) AS corr_with_log_life,
       round(abs(corr(f.value, log10(k.cycle_life))), 6) AS abs_corr
FROM early_features f
JOIN cells k ON f.cell_id = k.cell_id
WHERE k.split = 'train'
GROUP BY f.feature
ORDER BY abs_corr DESC, feature;
