import time
import numpy as np
from typing import Dict, Any

UINT32_MAX = 4294967295

def compute_search_reachability(graph) -> float:
    """
    Measures how many vertices can be reached from the graph's entry point via BFS.
    Returns reachability ratio (0.0 to 1.0).
    """
    graph_size = graph.size()
    edges_per_vertex = graph.get_edges_per_vertex()
    entry_vertices = graph.get_entry_vertex_indices()

    start_time = time.perf_counter()
    visited = np.zeros(graph_size, dtype=bool)
    frontier = []

    for s in entry_vertices:
        if 0 <= s < graph_size and not visited[s]:
            visited[s] = True
            frontier.append(s)

    while frontier:
        next_frontier = []
        for v in frontier:
            neighbors = graph.get_neighbor_indices(v)
            for e in range(edges_per_vertex):
                neighbor_index = neighbors[e]
                if neighbor_index != UINT32_MAX and neighbor_index < graph_size:
                    if not visited[neighbor_index]:
                        visited[neighbor_index] = True
                        next_frontier.append(neighbor_index)
        frontier = next_frontier

    count = int(np.sum(visited))
    elapsed_sec = time.perf_counter() - start_time
    print(f"Seed Reachability is {count} out of {graph_size} after {elapsed_sec:.2f}s")
    return count / max(graph_size, 1)

def compute_exploration_reach(graph) -> float:
    """
    Computes the average number of vertices reachable from any given vertex using flood fill
    with shortcut optimization.
    Returns average reachable fraction (0.0 to 1.0).
    """
    graph_size = graph.size()
    edges_per_vertex = graph.get_edges_per_vertex()
    start_time = time.perf_counter()

    best_vertex_reach = 0
    vertices_reach = []  # stores (vertex_id, reach_count, checked_ids_bitmap)
    index_of_vertex_reach = np.full(graph_size, graph_size, dtype=np.uint32)

    total_exploration_reachability = 0

    for entry_id in range(graph_size):
        checked_ids = np.zeros(graph_size, dtype=bool)
        check = [entry_id]
        checked_ids[entry_id] = True

        best_reach_vertex_index = 0
        best_reach_vertex_reach = 0

        while check and best_reach_vertex_reach < graph_size:
            check_next = []
            for check_index in check:
                neighbors = graph.get_neighbor_indices(check_index)
                for e in range(edges_per_vertex):
                    neighbor_index = neighbors[e]
                    if neighbor_index == UINT32_MAX or neighbor_index >= graph_size:
                        continue

                    if not checked_ids[neighbor_index]:
                        checked_ids[neighbor_index] = True
                        check_next.append(neighbor_index)

                        vertex_reach_idx = index_of_vertex_reach[neighbor_index]
                        if vertex_reach_idx < graph_size:
                            neighbor_reach_id, neighbor_reach_count, neighbor_reach_ids = vertices_reach[vertex_reach_idx]
                            if neighbor_reach_count == graph_size:
                                best_reach_vertex_index = vertex_reach_idx
                                best_reach_vertex_reach = graph_size
                                break

                            if neighbor_reach_count > best_reach_vertex_reach:
                                best_reach_vertex_reach = neighbor_reach_count
                                best_reach_vertex_index = vertex_reach_idx
                                checked_ids |= neighbor_reach_ids
                if best_reach_vertex_reach == graph_size:
                    break

            check = check_next

        if best_reach_vertex_reach == graph_size:
            index_of_vertex_reach[entry_id] = best_reach_vertex_index
            total_exploration_reachability += graph_size
        else:
            reach_count = int(np.sum(checked_ids))
            total_exploration_reachability += reach_count

            if best_vertex_reach < reach_count:
                best_vertex_reach = reach_count
                index_of_vertex_reach[entry_id] = len(vertices_reach)
                vertices_reach.append((entry_id, reach_count, checked_ids.copy()))
            elif best_reach_vertex_reach > 0:
                index_of_vertex_reach[entry_id] = best_reach_vertex_index
            else:
                index_of_vertex_reach[entry_id] = len(vertices_reach)
                vertices_reach.append((entry_id, reach_count, checked_ids.copy()))

    avg_reach = total_exploration_reachability / max(graph_size, 1)
    elapsed_sec = time.perf_counter() - start_time
    print(f"Average Exploration reachability is {avg_reach:.2f} after {elapsed_sec:.2f}s")
    return avg_reach / max(graph_size, 1)

def analyze_graph(graph) -> Dict[str, Any]:
    """
    Analyzes a search graph, computes all statistics (in/out degree, reachability, memory),
    and prints the output log matching C++ stats.h.
    """
    graph_size = graph.size()
    edges_per_vertex = graph.get_edges_per_vertex()
    try:
        dims = graph.get_feature_space().dim()
    except Exception:
        dims = 0

    print("\n--- Graph Analysis ---")
    
    # Compute out-degree stats & in-degree counts
    in_degree_count = np.zeros(graph_size, dtype=np.uint32)
    min_out = edges_per_vertex
    max_out = 0
    total_edges = 0

    for i in range(graph_size):
        neighbors = graph.get_neighbor_indices(i)
        valid_edges = 0
        for j in range(edges_per_vertex):
            n_idx = neighbors[j]
            if n_idx != UINT32_MAX and n_idx < graph_size:
                valid_edges += 1
                in_degree_count[n_idx] += 1
        
        total_edges += valid_edges
        if valid_edges < min_out:
            min_out = valid_edges
        if valid_edges > max_out:
            max_out = valid_edges

    avg_out = total_edges / max(graph_size, 1)

    # In-degree statistics
    min_in = int(np.min(in_degree_count)) if graph_size > 0 else 0
    max_in = int(np.max(in_degree_count)) if graph_size > 0 else 0
    avg_in = float(np.mean(in_degree_count)) if graph_size > 0 else 0.0
    source_vertices = int(np.sum(in_degree_count == 0)) if graph_size > 0 else 0

    # Memory estimation (vertex_count * (edges * 4 + weights * 4 + dims * 4))
    memory_bytes = graph_size * (edges_per_vertex * 4 + edges_per_vertex * 4 + dims * 4)

    print("Computing search reachability...")
    search_reach = compute_search_reachability(graph)

    print("Computing exploration reachability...")
    explore_reach = compute_exploration_reach(graph)

    print("Graph Statistics:")
    print(f"  Vertices: {graph_size}")
    print(f"  Total edges: {total_edges}")
    print(f"  Feature dimensions: {dims}")
    print(f"  Edges per vertex (k): {edges_per_vertex}")
    print(f"  Out-degree: avg={avg_out:.2f}, min={min_out}, max={max_out}")
    print(f"  In-degree:  avg={avg_in:.2f}, min={min_in}, max={max_in}, source_vertices={source_vertices}")
    print(f"  Search Reachability: {search_reach * 100:.2f}%")
    print(f"  Exploration Reachability: {explore_reach * 100:.2f}%")
    print(f"  Estimated memory: {memory_bytes / (1024.0 * 1024.0):.2f} MB")

    return {
        "vertex_count": graph_size,
        "edge_count": total_edges,
        "feature_dims": dims,
        "edges_per_vertex": edges_per_vertex,
        "avg_out_degree": avg_out,
        "min_out_degree": min_out,
        "max_out_degree": max_out,
        "avg_in_degree": avg_in,
        "min_in_degree": min_in,
        "max_in_degree": max_in,
        "source_vertices": source_vertices,
        "search_reachability": search_reach,
        "exploration_reachability": explore_reach,
        "memory_bytes": memory_bytes,
    }
