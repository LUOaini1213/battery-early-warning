-- ClickHouse DDL equivalent of sql/duckdb/00_schema.sql
CREATE TABLE IF NOT EXISTS cells (
    cell_id String, batch String, policy String,
    c1 Nullable(Float64), q1 Nullable(Float64), c2 Nullable(Float64), split String,
    cycle_life Float64, carried_over Bool
) ENGINE = MergeTree ORDER BY cell_id;

CREATE TABLE IF NOT EXISTS cycles (
    cell_id String, cycle Int32, qd Float64, qc Float64, ir Float64,
    tavg Float64, tmin Float64, tmax Float64, chargetime Float64
) ENGINE = MergeTree ORDER BY (cell_id, cycle);

CREATE TABLE IF NOT EXISTS early_features (cell_id String, feature String, value Float64)
ENGINE = MergeTree ORDER BY (feature, cell_id);

CREATE TABLE IF NOT EXISTS knee_labels (cell_id String, knee_onset Float64, knee_point Float64, eol Float64)
ENGINE = MergeTree ORDER BY cell_id;
