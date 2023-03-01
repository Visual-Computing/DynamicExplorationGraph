package com.vc.deg.graph;

/**
 * Iterates neighbors in a collection
 * 
 * @author Nico Hezel
 */
public interface NeighborConsumer {

    /**
     * Performs this operation on the given neighbor.
     * 
	 * @param id
	 * @param weight
	 */
    void accept(int id, float weight);
}
