package com.vc.deg;

import java.io.IOException;
import java.nio.file.Path;

public interface DynamicExplorationGraph {
	
	public GraphDesigner designer();
	public GraphNavigator navigator();
//	public int[] search(MemoryView query, int top);
//	public Node
	
	/**
     * Create an empty new graph
     * 
     * @param space
     * @return
     */
	public static DynamicExplorationGraph newGraph(FeatureSpace space, int edgesPerNode) {
		return GraphFactory.getDefaultFactory().newGraph(space, edgesPerNode);
	}
	
	/**
	 * Load an existing graph
	 * 
	 * @param file
	 * @return
	 */
	public static DynamicExplorationGraph loadGraph(Path file) throws ClassNotFoundException, IOException {
		return GraphFactory.getDefaultFactory().loadGraph(file);
	}
}