package com.vc.deg.graph;

import com.vc.deg.FeatureVector;

/**
 * A weighted undirected graph.
 * 
 * @author Nico Hezel
 */
public interface WeightedUndirectedGraph {

	/**
	 * Checks if a node with the given id already exists in the graph.
	 * 
	 * @param id
	 * @return
	 */
	public boolean hasNode(int id);
	
	/**
	 * Adds a node to the graph.
	 * Returns false if a node with the same id already exists.
	 * 
	 * @param id
	 * @param feature
	 * @return
	 */
	public boolean addNode(int id, FeatureVector feature);
	
	/**
	 * Removes a node from the graph.
	 * Returns false if no node with the id exists.
	 * 
	 * @param id
	 * @return
	 */
	public boolean removeNode(int id);
	
	/**
	 * A collection of all node ids in the graph
	 * 
	 * @return
	 */
	public NodeIds getNodeIds();
	
	
	/**
	 * Get the node data: label, feature and neighbors
	 * 
	 * @param id
	 * @return
	 */
	public NodeView getNode(int id);
	
	/**
	 * Does a directed edge between the two nodes exist?
	 * 
	 * @param id1
	 * @param id2
	 * @return
	 */
	public boolean hasEdge(int id1, int id2);
	
	/**
	 * Add a undirected edge with a given weight. 
	 * Return true if both of the edges existed before.
	 * 
	 * @param id1
	 * @param id2
	 * @param weight
	 * @return
	 */
	public boolean addUndirectedEdge(int id1, int id2, float weight);
	
	/**
	 * Remove a undirected edge between the two nodes.
	 * Return true if both of the edges existed before.
	 * 
	 * @param id1
	 * @param id2
	 * @return
	 */
	public boolean removeUndirectedEdge(int id1, int id2);
	
	/**
	 * Get the weight of the directed edge.
	 * Or 0 if the edes does not exists
	 * 
	 * @param id1
	 * @param id2
	 * @return
	 */
	public float getEdgeWeight(int id1, int id2);
	
	/**
	 * For a specific node retrieve the ids its connected nodes.
	 * 
	 * @param id
	 * @return
	 */
	public NodeIds getConnectedNodeIds(int id);
	
	/**
	 * Retrieve the edge information (id and weight) of connected edges for a specific node.
	 * 
	 * @param id
	 * @return
	 */
	public WeightedEdges getEdges(int id);
	
	
	/**
	 * Iterates all edges in the graph
	 * 
	 * @param consumer
	 */
	public void forEachEdge(WeightedEdgeConsumer consumer);
}
