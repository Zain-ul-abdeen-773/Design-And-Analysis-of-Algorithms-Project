# Dataset Notes

This project uses FIMI-style transaction files in `data/raw` and CSV mirrors in
`data/csv`.

Current local files:

| Dataset | Raw file | CSV file | Local transaction count | Status |
| --- | --- | --- | ---: | --- |
| Chess | `data/raw/chess.dat` | `data/csv/chess.csv` | 3,196 | Complete local copy |
| Connect | `data/raw/connect.dat` | `data/csv/connect.csv` | 67,557 | Complete local copy |
| Accidents | `data/raw/accidents.dat` | `data/csv/accidents.csv` | 35,809 | Incomplete local copy |

`scripts/dat_to_csv.py` regenerates the CSV mirrors from whatever `.dat` files
exist locally. The benchmark script reads the `.dat` files because FIMI datasets
are natively distributed as whitespace-separated transaction records.
