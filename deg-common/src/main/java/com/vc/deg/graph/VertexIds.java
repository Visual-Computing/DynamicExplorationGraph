package com.vc.deg.graph;

import java.util.function.IntConsumer;

/**
 * An immutable collection with node ids.
 * 
 * @author Nico Hezel
 */
public interface VertexIds extends Iterable<Integer> {

	/**
	 * Number of node ids in this collection
	 * 
	 * @return
	 */
	public int size();
	
	/**
	 * Does this collection contains the node id
	 * 
	 * @param id
	 * @return
	 */
	public boolean contains(int id);
	
	/**
	 * For every node id in this collection the consumer is called.
	 * 
	 * @param consumer
	 */
	public void forEach(IntConsumer consumer);

	/**
	 * Creates a copy of all the node ids in this collection
	 * and stores them in an int-array. Any changes to this
	 * array to not effect the collection.
	 * 
	 * @return
	 */
	public int[] toArray();
	
}
