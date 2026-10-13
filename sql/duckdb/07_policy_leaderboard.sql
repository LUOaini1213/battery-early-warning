-- 07 Charging-policy leaderboard: cycle-life statistics per fast-charging policy.
SELECT policy, count(*) AS n_cells, round(avg(cycle_life), 2) AS mean_life,
       min(cycle_life) AS min_life, max(cycle_life) AS max_life
FROM cells
GROUP BY policy
ORDER BY mean_life DESC, policy;
