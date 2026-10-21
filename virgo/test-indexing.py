import sqlite3
conn = sqlite3.connect("/mnt/scratch/baburish/doublepulse/gnn/Analysis/data/combined_nutau_5TeV.db")
plan = conn.execute("""
EXPLAIN QUERY PLAN
SELECT p.event_no, p.dom_x, p.dom_y, p.dom_z, p.dom_time, p.charge,
       r.cascade_vertex_fit_x, r.cascade_vertex_fit_y, r.cascade_vertex_fit_z,
       r.cascade_reco_energy_tev
FROM   CleanedROIPulses p
JOIN   truth t ON p.event_no = t.event_no
JOIN   reco  r ON p.event_no = r.event_no
ORDER  BY p.event_no
""").fetchall()
for row in plan:
    print(row)
conn.close()
