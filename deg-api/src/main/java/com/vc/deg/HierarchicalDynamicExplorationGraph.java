package com.vc.deg;

import java.io.IOException;
import java.nio.file.Path;

public interface HierarchicalDynamicExplorationGraph {

	public GraphDesigner designer();
	
	/**
	 * Stores the graph structural data and the feature vectors into a file.
	 * It includes the FeatureSpace, Nodes, Edges, Features, Labels but not 
	 * information about the Design process or Navigation settings.
	 * 
	 * @param file
	 * @throws ClassNotFoundException
	 * @throws IOException
	 */
	public void writeToFile(Path file) throws ClassNotFoundException, IOException;
	
	/**
	 * Search the graph for the best nodes matching the query
	 * 
	 * @param query
	 * @param k
	 * @return
	 */
	public default SearchResult search(FeatureVector query, int atLevel, int k) {
		return search(query, atLevel, k, 0.1f);
	}
	
	/**
	 * 
	 * @param query
	 * @param k
	 * @param eps Is similar to a search radius factor 0 means low and 1 means high radius to scan
	 * @return
	 */
	public SearchResult search(FeatureVector query, int atLevel, int k, float eps);
	
	
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
	
	/**
	 * Load an existing graph
	 * 
	 * @param file
	 * @return
	 */
	public static HierarchicalDynamicExplorationGraph loadGraph(Path file, String componentType) throws ClassNotFoundException, IOException {
		return GraphFactory.getDefaultFactory().loadHierchicalGraph(file, componentType);
	}
}
