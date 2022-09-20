package com.vc.deg.ref;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import com.vc.deg.DynamicExplorationGraph;
import com.vc.deg.FeatureSpace;
import com.vc.deg.FeatureVector;
import com.vc.deg.GraphDesigner;
import com.vc.deg.GraphFactory;
import com.vc.deg.SearchResult;
import com.vc.deg.ref.designer.EvenRegularGraphDesigner;
import com.vc.deg.ref.graph.MapBasedWeightedUndirectedRegularGraph;
import com.vc.deg.ref.hierarchy.HierarchicalGraphDesigner;
import com.vc.deg.ref.navigation.MapBasedGraphNavigator;

public class HierarchicalDynamicExplorationGraph implements com.vc.deg.HierarchicalDynamicExplorationGraph {

	protected final List<DynamicExplorationGraph> layers;
	protected final HierarchicalGraphDesigner designer;
	
	/**
	 * Define how the nodes of the graph will be compared and how many edges each node has.
	 * 
	 * @param space
	 * @param edgesPerNode
	 */
	public HierarchicalDynamicExplorationGraph(FeatureSpace space, int edgesPerNode, int topRankSize) {
		this.layers = new ArrayList<>();
		this.designer = null;
	}
	
	/**
	 * Used by the copy constructor and read file methods
	 * 
	 * @param graph
	 */
	private HierarchicalDynamicExplorationGraph(List<DynamicExplorationGraph> layers) {
		this.layers = layers;
		this.designer = null;
	}
	
	@Override
	public HierarchicalGraphDesigner designer() {
		return designer;
	}

	/**
	 * 
	 * @param query
	 * @param k
	 * @param eps Is similar to a search radius factor 0 means low and 1 means high radius to scan
	 * @return
	 */
	@Override
	public SearchResult search(FeatureVector query, int atLevel, int k, float eps) {
		return null;
	}
	
	
	/**
	 * Stores the graph structural data and the feature vectors into a file.
	 * It includes the FeatureSpace, Nodes, Edges, Features, Labels but not 
	 * information about the Design process or Navigation settings.
	 * 
	 * @param file
	 * @throws ClassNotFoundException
	 * @throws IOException
	 */
	@Override
	public void writeToFile(Path file) throws ClassNotFoundException, IOException {
		
	}
	
	/**
	 * Read meta data for the {@link FeatureSpace}, number of nodes and edges
	 * 
	 * @param file
	 * @return
	 * @throws ClassNotFoundException
	 * @throws IOException
	 */
	public static HierarchicalDynamicExplorationGraph readFromFile(Path file) throws IOException {
		return new HierarchicalDynamicExplorationGraph(null);
	}
	
	public static HierarchicalDynamicExplorationGraph readFromFile(Path file, String featureType) throws IOException {
		return new HierarchicalDynamicExplorationGraph(null);
	}
}
