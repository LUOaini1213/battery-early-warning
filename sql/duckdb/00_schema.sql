-- DuckDB schema (tables are filled from pandas by bew/sql_layer.py)
CREATE TABLE cells (
    cell_id VARCHAR PRIMARY KEY, batch VARCHAR, policy VARCHAR,
    c1 DOUBLE, q1 DOUBLE, c2 DOUBLE, split VARCHAR,
    cycle_life DOUBLE, carried_over BOOLEAN
);
CREATE TABLE cycles (
    cell_id VARCHAR, cycle INTEGER, qd DOUBLE, qc DOUBLE, ir DOUBLE,
    tavg DOUBLE, tmin DOUBLE, tmax DOUBLE, chargetime DOUBLE
);
CREATE TABLE early_features (cell_id VARCHAR, feature VARCHAR, value DOUBLE);
CREATE TABLE knee_labels (cell_id VARCHAR, knee_onset DOUBLE, knee_point DOUBLE, eol DOUBLE);
