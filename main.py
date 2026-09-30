"""
Stage 18 — full autonomous multi-cycle research loop.

    ROOT TOPIC
        -> N research cycles (planner, search, download, chunk, embed,
           retrieve, analyze, dedup, next topic)
        -> research trajectory printed as a tree

Usage:
    python main.py                      # asks for topic, MAX_CYCLES cycles
    python main.py 2 "Open quantum systems"   # custom cycles + topic
"""

import sys

from src.controller.cycle import ResearchController, render_tree
from src.embeddings.embedder import Embedder
from src.topic.planner import Topic

MAX_CYCLES = 3


def main() -> None:
    max_cycles = int(sys.argv[1]) if len(sys.argv) > 1 else MAX_CYCLES
    topic_name = (
        sys.argv[2] if len(sys.argv) > 2
        else input("Enter your topic: ").strip()
    )

    controller = ResearchController()

    root = Topic(name=topic_name, parent=None, cycle=0)
    controller.memory.add_topic(
        root, controller.embedder.embed_texts([root.name])[0]
    )

    topic = root
    for cycle_index in range(max_cycles):
        next_topic = controller.run_cycle(topic, cycle_index)
        if next_topic is None:
            print("\n[loop] stopping: no novel topic to explore")
            break
        topic = next_topic

    print(f"\n{'=' * 70}\nRESEARCH TRAJECTORY\n{'=' * 70}")
    print(render_tree(controller.memory))


if __name__ == "__main__":
    main()
