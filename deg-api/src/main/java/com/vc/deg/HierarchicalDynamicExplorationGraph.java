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
	
	/**
	 * Search the graph for the best vertices matching the query at the given hierarchy level.
	 * 
	 * @param query
	 * @param k
	 * @return
	 */
	public default int[] search(FeatureVector query, int atLevel, int k) {
		return search(query, atLevel, k, 0.1f);
	}
	
	@Override
	public default int[] search(FeatureVector query, int k, float eps) {
		return search(query, 0, k, 0.1f);
	}
	
	/**
	 * Search the graph for the best vertices matching the query at the given hierarchy level.
	 * 
	 * @param query
	 * @param k
	 * @param eps Is similar to a search radius factor
	 * @return
	 */
	public int[] search(FeatureVector query, int atLevel, int k, float eps);
	
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
