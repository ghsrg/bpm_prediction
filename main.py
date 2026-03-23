import sys
from src.cli import main as train_main
from tools.visualize_topology import main as visualize_topology_main
from tools.visualize_graph import main as visualize_graph_main
from tools.ingest_topology import main as ingest_topology_main
from tools.sync_topology import main as sync_topology_main
from tools.sync_stats import main as sync_stats_main
from tools.sync_stats_backfill import main as sync_stats_backfill_main
from tools.experiment_ui import main as experiment_ui_main

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ingest-topology":
        sys.exit(ingest_topology_main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "visualize-topology":
        sys.exit(visualize_topology_main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "visualize-graph":
        sys.exit(visualize_graph_main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "sync-topology":
        sys.exit(sync_topology_main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "sync-stats":
        sys.exit(sync_stats_main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "sync-stats-backfill":
        sys.exit(sync_stats_backfill_main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "experiment-ui":
        sys.exit(experiment_ui_main(sys.argv[2:]))
    sys.exit(train_main())
