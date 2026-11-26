import os
import pandas as pd
import structure_pipeline
import lava_pipeline
import route_optimizer
from valid_path import valid_paths

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BIOME_CSV_PATH = os.path.join(BASE_DIR, "biome_cost_lookup.csv")
NETHER_CSV_PATH = os.path.join(BASE_DIR, "nether_biome_cost_lookup.csv")

SERVER_PATH = r"D:\LavaServer"
OUTPUT_DIR = os.path.join(SERVER_PATH, "seeds")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_master_graph(seed: int, radius_blocks: int):
    print(f"\n{'=' * 50}\n ROZPOCZYNAM ANALIZĘ GRAFU DLA SEEDU {seed}\n{'=' * 50}")

    # 1. Zbieranie struktur z Cubiomes
    struct_df = structure_pipeline.get_structures_df(seed, radius_blocks)

    # 2. Zbieranie lawy z serwera Fabric
    lava_raw_df = lava_pipeline.get_lava_df(seed, radius_blocks // 16, SERVER_PATH)

    lava_df = pd.DataFrame()
    if not lava_raw_df.empty:
        lava_df = pd.DataFrame({
            "node_type": "Lava_Pool",
            "x": lava_raw_df["x"],
            "z": lava_raw_df["z"],
            "dimension": "Overworld",
            "weight": lava_raw_df["block_count"]
        })

    # 3. Dodanie punktu startowego (prawdziwy spawn z Cubiomes)
    spawn_x, spawn_z = structure_pipeline.get_world_spawn(seed)
    spawn_df = pd.DataFrame([
        {"node_type": "Spawn", "x": spawn_x, "z": spawn_z, "dimension": "Overworld", "weight": 0.0}
    ])

    # 4. Agregacja i zapis grafu do CSV
    master_df = pd.concat([spawn_df, struct_df, lava_df], ignore_index=True)
    master_df.insert(0, 'node_id', master_df.index)

    final_csv_path = os.path.join(OUTPUT_DIR, f'master_graph_{seed}.csv')
    with open(final_csv_path, 'w', encoding='utf-8') as f:
        f.write(f"SEED: {seed}\n")
        master_df.to_csv(f, index=False)

    print("\n--- GOTOWA TABELA WĘZŁÓW ---")
    print(master_df.to_string(index=False))
    print("-" * 28)
    print(f"Zapisano czysty plik: {final_csv_path}\n")

    # 5. PEŁNA OPTYMALIZACJA (OVERWORLD + PORTAL 1:8 + NETHER + PORTAL 8:1)
    ow_map, ow_min = route_optimizer.load_overworld_cost_model(BIOME_CSV_PATH)
    net_map, net_min = route_optimizer.load_nether_cost_model(NETHER_CSV_PATH)

    ow_sampler = structure_pipeline.create_biome_sampler(seed, structure_pipeline.Dimension.DIM_OVERWORLD)
    net_sampler = structure_pipeline.create_biome_sampler(seed, structure_pipeline.Dimension.DIM_NETHER)

    ow_opt = route_optimizer.RouteOptimizer(ow_map, ow_min, ow_sampler, step=32)
    net_opt = route_optimizer.RouteOptimizer(net_map, net_min, net_sampler, step=16, fallback_cost=4.50)

    nodes_by_type = {
        ntype: group.to_dict("records")
        for ntype, group in master_df.groupby("node_type")
    }

    best_run = None
    for scenario in valid_paths:
        result = route_optimizer.evaluate_full_scenario(nodes_by_type, scenario, ow_opt, net_opt)
        if result:
            print(f"Scenariusz {scenario}: {result['total_time_sec']} s")
            if best_run is None or result["total_time_sec"] < best_run["total_time_sec"]:
                best_run = result

    if best_run:
        print(f"\n>>> NAJLEPSZA TRASA CAŁKOWITA: {best_run['total_time_sec']} s")
        print(f"Zwycięski scenariusz: {best_run['scenario']}")
        for n in best_run["chosen_sequence"]:
            print(f" -> [{n['dimension']}] {n['node_type']} na ({n['x']}, {n['z']})")
    else:
        print("\n[UWAGA] Brak kompletnego zestawu struktur dla żadnego ze scenariuszy.")


if __name__ == '__main__':
    TEST_SEED = 2115
    RADIUS_IN_BLOCKS = 1024

    generate_master_graph(TEST_SEED, RADIUS_IN_BLOCKS)
