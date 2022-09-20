package com.vc.deg.ref;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Iterator;
import java.util.TreeSet;

import com.vc.deg.FeatureSpace;
import com.vc.deg.FeatureVector;
import com.vc.deg.SearchResult;
import com.vc.deg.SearchResult.SearchEntry;
import com.vc.deg.ref.designer.EvenRegularGraphDesigner;
import com.vc.deg.ref.graph.MapBasedWeightedUndirectedRegularGraph;
import com.vc.deg.ref.navigation.MapBasedGraphNavigator;
import com.vc.deg.ref.search.ObjectDistance;
import com.vc.deg.ref.search.ResultSet;


/**
 * This class wraps the main functions of the library
 * 
 * TODO copy over the content of MapBasedWeightedUndirectedRegularGraph
 * 
 * @author Nico Hezel
 */
public class DynamicExplorationGraph implements com.vc.deg.DynamicExplorationGraph {

	protected final MapBasedWeightedUndirectedRegularGraph internalGraph;
	protected final MapBasedGraphNavigator navigator;
	protected final EvenRegularGraphDesigner designer;

	/**
	 * Define how the nodes of the graph will be compared and how many edges each node has.
	 * 
	 * @param space
	 * @param edgesPerNode
	 */
	public DynamicExplorationGraph(FeatureSpace space, int edgesPerNode) {
		this.internalGraph = new MapBasedWeightedUndirectedRegularGraph(edgesPerNode, space);
		this.designer = new EvenRegularGraphDesigner(internalGraph); 
		this.navigator = new MapBasedGraphNavigator(internalGraph); 
	}

	/**
	 * Define how the nodes of the graph will be compared and how many edges each node has.
	 * 
	 * @param space
	 * @param expectedSize
	 * @param edgesPerNode
	 */
	public DynamicExplorationGraph(FeatureSpace space, int expectedSize, int edgesPerNode) {
		this.internalGraph = new MapBasedWeightedUndirectedRegularGraph(edgesPerNode, expectedSize, space);
		this.designer = new EvenRegularGraphDesigner(internalGraph); 
		this.navigator = new MapBasedGraphNavigator(internalGraph); 
	}
	
	/**
	 * Used by the copy constructor and read file methods
	 * 
	 * @param graph
	 */
	private DynamicExplorationGraph(MapBasedWeightedUndirectedRegularGraph graph) {
		this.internalGraph = graph;
		this.designer = new EvenRegularGraphDesigner(internalGraph); 
		this.navigator = new MapBasedGraphNavigator(internalGraph); 
	}

	@Override
	public MapBasedGraphNavigator navigator() {
		return navigator;
	}

	@Override
	public EvenRegularGraphDesigner designer() {
		return designer;
	}
	
	@SuppressWarnings("unchecked")
	@Override
	public SearchResult search(FeatureVector query, int k, float eps) {
		final int[] forbiddenIds = new int[0];
		final int[] entryPoint = new int[] { internalGraph.getVertices().iterator().next().getId() };
		final TreeSet<? extends SearchEntry> result = internalGraph.search(query, k, eps, forbiddenIds, entryPoint);
		return () -> (Iterator<SearchEntry>) result.iterator();
		
//		TreeSet<ObjectDistance> result = internalGraph.search(query, k, eps, forbiddenIds, entryPoint);
//		return new ResultSet(result);
	}


	/**
	 * Goal: read and write should be compatible with deglib in c-lang. Problem is the endians and the FeatureVector object type written into the output file.
	 * 		 We might want to store another file next to it with the FeatureVector object type information (e.g. storing a single FeatureVector object).
	 * 		 - we use the FeatureVector and memory pool option to when reading from file 
	 * 		 - we could write primitive types in the filename (good for c-lang too) 
	 * 		 - if type in filename is object, we ask FeatureFactory.getCustomFactory()
	 * 
	 * 
	 * @param file
	 * @throws ClassNotFoundException
	 * @throws IOException
	 */
	@Override
	public void writeToFile(Path file) throws ClassNotFoundException, IOException {
		this.internalGraph.writeToFile(file);
	}

	/**
	 * Read meta data for the {@link FeatureSpace}, number of nodes and edges
	 * 
	 * @param file
	 * @return
	 * @throws ClassNotFoundException
	 * @throws IOException
	 */
	public static DynamicExplorationGraph readFromFile(Path file) throws IOException {
		return new DynamicExplorationGraph(MapBasedWeightedUndirectedRegularGraph.readFromFile(file));
	}
	
	public static DynamicExplorationGraph readFromFile(Path file, String featureType) throws IOException {
		return new DynamicExplorationGraph(MapBasedWeightedUndirectedRegularGraph.readFromFile(file, featureType));
	}
}