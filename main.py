import sys
from src.cli import main as train_main
from tools.visualize_topology import main as visualize_topology_main
from tools.visualize_graph import main as visualize_graph_main
from tools.ingest_topology import main as ingest_topology_main

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ingest-topology":
        sys.exit(ingest_topology_main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "visualize-topology":
        sys.exit(visualize_topology_main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "visualize-graph":
        sys.exit(visualize_graph_main(sys.argv[2:]))
    sys.exit(train_main())
