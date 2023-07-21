package com.vc.deg.graph;

import com.vc.deg.FeatureVector;

/**
 * Goal: All get methods (id, feature, neighbor count, neighbor id at index, neighbor weight at index) can be lazy. 
 * 		 The user of the API does not have problems with the lambda scopes. 
 * 
 * 
 * TODO Add a position(), position(int newPos) and size() Method. 
 * 		The reference implementation would need to create an array the first time position(int newPos) is called.
 * 		For a readonly graph the reference implementation should already have such an array.
 * 
 * @author Nico Hezel
 *
 */
public interface VertexCursor {

	/**
	 * Move to the next vertex. Need to be
	 * 
	 * @return
	 */
	public boolean moveNext();
	
	/**
	 * Label of the current vertex
	 * {@link #moveNext()} need to be called before
	 * 
	 * @return
	 */
	public int getVertexLabel();
	
	/**
	 * Feature vector of the current vertex
	 * {@link #moveNext()} need to be called before
	 * 
	 * @return
	 */
	public FeatureVector getVertexFeature();
	
	/**
	 * Neighbor labels and weights of the current vertex
	 * {@link #moveNext()} need to be called before
	 * 
	 * @param neighborConsumer
	 */
	public void forEachNeighbor(NeighborConsumer neighborConsumer);
}
