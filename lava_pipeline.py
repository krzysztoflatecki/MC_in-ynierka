import subprocess
import time
import os
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def set_server_seed(server_dir, new_seed):
    prop_path = os.path.join(server_dir, "server.properties")
    with open(prop_path, "r") as file: lines = file.readlines()
    with open(prop_path, "w") as file:
        for line in lines:
            file.write(f"level-seed={new_seed}\n" if line.startswith("level-seed=") else line)


def get_lava_df(seed: int, radius_chunks: int, server_dir: str) -> pd.DataFrame:
    """Odpala serwer, pobiera lawę, klasteryzuje, usuwa plik tymczasowy i zwraca DataFrame."""
    print(f"[{seed}] Uruchamianie serwera Javy (ekstrakcja lawy)...")
    set_server_seed(server_dir, seed)

    world_dir = os.path.join(server_dir, "world")
    if os.path.exists(world_dir):
        os.system(f'rmdir /s /q "{world_dir}"')

    process = subprocess.Popen(
        ["java", f"-Dextract.radius={radius_chunks}", "-jar", "fabric-server-launch.jar", "nogui"],
        cwd=server_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    process.wait()

    raw_csv_path = os.path.join(server_dir, f"lava_pools_{seed}.csv")
    final_df = pd.DataFrame()

    # Przetwarzanie i sprzątanie
    if os.path.exists(raw_csv_path):
        raw_df = pd.read_csv(raw_csv_path)
        os.remove(raw_csv_path)  # Usunięcie pliku tymczasowego

        if not raw_df.empty:
            coords = raw_df[['x', 'y', 'z']].to_numpy()
            db = DBSCAN(eps=1.75, min_samples=1, metric='euclidean').fit(coords)
            raw_df = raw_df.copy()
            raw_df['pool_id'] = db.labels_

            pools = raw_df.groupby('pool_id').agg(
                x=('x', lambda s: int(np.round(s.mean()))),
                z=('z', lambda s: int(np.round(s.mean()))),
                block_count=('x', 'count')
            ).reset_index(drop=True)

            final_df = pools[pools['block_count'] >= 3].copy()

    return final_df
