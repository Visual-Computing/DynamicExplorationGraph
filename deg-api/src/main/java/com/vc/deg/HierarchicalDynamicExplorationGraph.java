package com.vc.deg;

import java.io.IOException;
import java.nio.file.Path;

public interface HierarchicalDynamicExplorationGraph extends DynamicExplorationGraph {

	public GraphDesigner designer();
	
	/**
	 * Stores each layer of the graph into a separate file.
	 * The target directory will be created if it does not exists.
	 * 
	 * @param targetDir
	 * @throws ClassNotFoundException
	 * @throws IOException
	 */
	@Override
	public void writeToFile(Path targetDir) throws ClassNotFoundException, IOException;
	
	@Override
	public default int[] search(FeatureVector query, int k, float eps, int[] forbiddenIds) {
		return search(query, 0, k, 0.1f, forbiddenIds);
	}
	
	/**
	 * Search the graph for the best vertices matching the query at the given hierarchy level.
	 * 
	 * @param query
	 * @param k
	 * @return
	 */
	public default int[] search(FeatureVector query, int atLevel, int k) {
		return search(query, atLevel, k, 0.1f, new int[0]);
	}
	
	/**
	 * Search the graph for the best vertices matching the query at the given hierarchy level.
	 * 
	 * @param query
	 * @param atLevel
	 * @param k
	 * @param eps Is similar to a search radius factor
	 * @param forbiddenIds
	 * @return
	 */
	public int[] search(FeatureVector query, int atLevel, int k, float eps, int[] forbiddenIds);
	
	@Override
	default int[] explore(int entryLabel, int k, int maxDistanceComputationCount, int[] forbiddenIds) {
		return explore(entryLabel, 0, k, maxDistanceComputationCount, forbiddenIds);
	}
	
	/**
	 * Start from the entry vertex at the given hierarchical layer and explore the neighborhood to find k-similar neighbors.
	 * The number of distance calculations can be limited.
	 * 
	 * @param entryLabel
	 * @param atLevel
	 * @param k
	 * @param maxDistanceComputationCount
	 * @return
	 */
	public int[] explore(int entryLabel, int atLevel, int k, int maxDistanceComputationCount, int[] forbiddenIds);
	
	
	/**
	 * Create a copy of the graph
	 * 
	 * @return
	 */
	@Override
	public HierarchicalDynamicExplorationGraph copy();
	
	/**
     * Create an empty new graph
     * 
     * @param space
     * @return
     */
	public static HierarchicalDynamicExplorationGraph newGraph(FeatureSpace space, int edgesPerNode, int topRankSize) {
		return GraphFactory.getDefaultFactory().newHierchicalGraph(space, edgesPerNode, topRankSize);
	}
	
	/**
	 * Load an existing graph
	 * 
	 * @param file
	 * @return
	 */
	public static HierarchicalDynamicExplorationGraph loadGraph(Path file) throws ClassNotFoundException, IOException {
		return GraphFactory.getDefaultFactory().loadHierchicalGraph(file);
	}
}
