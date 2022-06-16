package com.vc.deg.graph;

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
