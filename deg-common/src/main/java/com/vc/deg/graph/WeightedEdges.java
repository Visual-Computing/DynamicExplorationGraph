package com.vc.deg.graph;

import java.util.function.IntConsumer;

/**
 * An immutable collection of weighted edges for a specific node.
 * The edge weights and ids of the connected other nodes can be retrieved.
 * 
 * @author Nico Hezel
 *
 */
public interface WeightedEdges extends VertexIds {
	
	/**
	 * The id of the node where the edges belong to.
	 * 
	 * @return
	 */
	public int getNodeId();
	
	/**
	 * Number of edges in this collection
	 * 
	 * @return
	 */
	public int size();
	
	/**
	 * Does this collection contain an edge to a specific node
	 * 
	 * @param id
	 * @return
	 */
	public boolean contains(int id);
	
	/**
	 * For every weighed edge in this collection the consumer is called.
	 * Just the ids of the connected nodes is presented.
	 * See {@link #forEach(WeightedEdgeConsumer)} for more edge weights.
	 * 
	 * @param consumer
	 */
	public void forEach(IntConsumer consumer);
	
	/**
	 * For every weighed edge in this collection the consumer is called.
	 * 
	 * @param consumer
	 */
	public void forEach(WeightedEdgeConsumer consumer);
	
}
