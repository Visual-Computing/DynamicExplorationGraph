package com.vc.deg.impl.graph;

import java.util.function.IntConsumer;

/**
 * An immutable collection of weighted edges for a specific node.
 * The edge weights and ids of the connected other nodes can be retrieved.
 * 
 * @author Nico Hezel
 *
 */
public interface WeightedEdges {
	
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
	
	/**
	 * Provides information about edges.
	 * 
	 * @author Nico Hezel
	 */
	public interface WeightedEdgeConsumer {

		/**
		 * Provides information about a single edge,
		 * every time this method is called.
		 * 
		 * @param id1
		 * @param id2
		 * @param weight
		 */
	    void accept(int id1, int id2, float weight);
	    
	}
}