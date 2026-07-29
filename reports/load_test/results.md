| Containers | Users | Requests | Failures | RPS | p50 (ms) | p95 (ms) | p99 (ms) | Replicas hit |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 1,815 | 0 | 45.38 | 100 | 340 | 550 | 1 |
| 1 | 50 | 2,434 | 1 | 60.85 | 320 | 1700 | 2200 | 1 |
| 1 | 100 | 2,472 | 0 | 61.8 | 630 | 4200 | 6800 | 1 |
| 2 | 20 | 2,131 | 1 | 53.27 | 43 | 190 | 280 | 2 |
| 2 | 50 | 4,035 | 1 | 100.88 | 120 | 580 | 910 | 2 |
| 2 | 100 | 4,151 | 0 | 103.78 | 320 | 3000 | 3900 | 2 |
| 4 | 20 | 1,962 | 0 | 63.29 | 35 | 210 | 420 | 4 |
| 4 | 50 | 5,375 | 0 | 134.38 | 28 | 120 | 200 | 4 |
| 4 | 100 | 6,084 | 2 | 152.1 | 170 | 1300 | 2200 | 4 |

RPS is computed as requests / measured elapsed time. Locust's own `Requests/s` column reports the *instantaneous* rate at the final snapshot and is unreliable -- it read 0.94 for a run that served 1,962 requests in 45 s.
