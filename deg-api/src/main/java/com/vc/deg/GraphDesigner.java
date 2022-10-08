package com.vc.deg;

import java.util.Random;
import java.util.function.IntPredicate;

/**
 * Adds or removed graph nodes and improves the connections. 
 * 
 * @author Nico Hezel
 *
 */
public interface GraphDesigner {
	
	/**
	 * Set the random generator for all operations
	 * 
	 * @param rnd
	 */
	public void setRandom(Random rnd);
	
	/**
	 * Queue new data to add to the graph
	 * 
	 * @param label
	 * @param data
	 */
	public void add(int label, FeatureVector data);
	
	/**
	 * Queue the removal of an existing node
	 * 
	 * @param label
	 */
	public void remove(int label);
	
	/**
	 * Queue the removal of all vertices which are in the graph but do not pass the filter
	 * 
	 * @param filter
	 */
	public void removeIf(IntPredicate filter);	
	
	/**
	 * Builds and improve the graph. All new data points or removal requests will be processed first
	 * afterwards the edges of the existing graph will be improved infinitely.
	 * 
	 * @param listener will be informed about every change
	 */
	public void build(ChangeListener listener);
	
	/**
	 * Stops the current building process after the next change.
	 */
	public void stop();
	
	
	/**
	 * The listener will be informed about every change
	 * 
	 * @author Nico Hezel
	 */
	public static interface ChangeListener {

		/**
		 * The method gets called after every change: 
		 * - adding a node
		 * - removing a node
		 * - improving a path of edges
		 * 
		 * @param step number of graph manipulation steps
		 * @param added number of added nodes
		 * @param deleted number of deleted nodes
		 * @param improved number of successful improvement
		 * @param tries number of improvement tries
		 * @param lastAdd label of the vertex added in this change
		 * @param lastDelete label of the vertex deleted in this change
		 */
		public void onChange(long step, long added, long deleted, long improved, long tries, int lastAdd, int lastDelete);
	}

	
	
	// ----------------------------------------------------------------------------------------------
	// ----------------------------------- hyper parameters -----------------------------------------
	// ----------------------------------------------------------------------------------------------
	
	/**
	 * Hyper parameter when adding new vertices
	 * In order to find good candidates a search is performed, an this K is the search k.
	 * 
	 * @param k
	 */
	public void setExtendK(int k);

	/**
	 * Hyper parameter when adding new vertices
	 * In order to find good candidates a search is performed, an this eps is the search eps.
	 * 
	 * @param eps
	 */
	public void setExtendEps(float eps);
	
	/**
	 * Hyper parameter when improving the edges
	 * In order to find good candidates a search is performed, an this K is the search k.
	 * 
	 * @param k
	 */
	public void setImproveK(int k);

	/**
	 * Hyper parameter when improving the edges
	 * In order to find good candidates a search is performed, an this eps is the search eps.
	 * 
	 * @param eps
	 */
	public void setImproveEps(float eps);
	
	/**
	 * Hyper parameter when improving the edges
	 * 
	 * @param maxPathLength
	 */
	public void setMaxPathLength(int maxPathLength);

	
	
	// ----------------------------------------------------------------------------------------------
	// ----------------------------------- evaluation methods ---------------------------------------
	// ----------------------------------------------------------------------------------------------
	
	/**
	 * Compute the average edge weight of all edges.
	 * @return
	 */
	public float calcAvgEdgeWeight();
	
	/**
	 * Check if all vertices have a specific amount of neighbors and their edge weight corresponds to the
	 * distance to the neighbor.
	 * 
	 * @param expectedVertices
	 * @param expectedNeighbors
	 * @return
	 */
	public boolean checkGraphValidation(int expectedVertices, int expectedNeighbors);
}
