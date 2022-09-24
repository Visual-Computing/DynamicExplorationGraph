package com.vc.deg.ref;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Iterator;
import java.util.TreeSet;
import java.util.function.BiConsumer;

import com.vc.deg.FeatureSpace;
import com.vc.deg.FeatureVector;
import com.vc.deg.ref.designer.EvenRegularGraphDesigner;
import com.vc.deg.ref.graph.MapBasedWeightedUndirectedRegularGraph;
import com.vc.deg.ref.search.ObjectDistance;


/**
 * This class wraps the main functions of the library
 * 
 * @author Nico Hezel
 */
public class DynamicExplorationGraph implements com.vc.deg.DynamicExplorationGraph {

	protected final MapBasedWeightedUndirectedRegularGraph internalGraph;
	protected final EvenRegularGraphDesigner designer;

	/**
	 * Define how the vertices of the graph will be compared and how many edges each vertex has.
	 * 
	 * @param space
	 * @param edgesPerVertex
	 */
	public DynamicExplorationGraph(FeatureSpace space, int edgesPerVertex) {
		this.internalGraph = new MapBasedWeightedUndirectedRegularGraph(edgesPerVertex, space);
		this.designer = new EvenRegularGraphDesigner(internalGraph); 
	}

	/**
	 * Define how the vertices of the graph will be compared and how many edges each vertex has.
	 * 
	 * @param space
	 * @param expectedSize
	 * @param edgesPerVertex
	 */
	public DynamicExplorationGraph(FeatureSpace space, int expectedSize, int edgesPerVertex) {
		this.internalGraph = new MapBasedWeightedUndirectedRegularGraph(edgesPerVertex, expectedSize, space);
		this.designer = new EvenRegularGraphDesigner(internalGraph); 
	}
	
	/**
	 * Used by the copy constructor and read file methods
	 * 
	 * @param graph
	 */
	private DynamicExplorationGraph(MapBasedWeightedUndirectedRegularGraph graph) {
		this.internalGraph = graph;
		this.designer = new EvenRegularGraphDesigner(internalGraph); 
	}
	
	
	
	/**
	 * Iterate all vertices and call the consumer with every vertex id
	 * 
	 * @return
	 */
	public void forEachVertex(BiConsumer<Integer, FeatureVector> consumer) {
		this.internalGraph.getVertices().forEach(vertex -> consumer.accept(vertex.getId(), vertex.getFeature()));
	}
	
	/**
	 * Does this vertex exists?
	 * 
	 * @param label
	 * @return
	 */
	public boolean hasVertex(int label) {
		return this.internalGraph.hasVertex(label);
	}

	/**
	 * 
	 * @param id
	 * @return
	 */
	public FeatureVector getFeature(int id) {
		return this.internalGraph.getVertex(id).getFeature();
	}
	
	/**
	 * 
	 * @return
	 */
	public Iterator<Integer> getVertexIds() {
		return this.internalGraph.getVertexIds().iterator();
	}

	/**
	 * Number of vertices
	 * 
	 * @return
	 */
	public int size() {
		return this.internalGraph.getVertexCount();
	}
	
	
	@Override
	public EvenRegularGraphDesigner designer() {
		return designer;
	}
	
	@Override
	public int[] search(FeatureVector query, int k, float eps, int[] forbiddenIds) {
		final int[] entryPoint = new int[] { internalGraph.getVertices().iterator().next().getId() };
		final TreeSet<ObjectDistance> topList = internalGraph.search(query, k, eps, forbiddenIds, entryPoint);
		
		final int[] result = new int[topList.size()];
		final Iterator<ObjectDistance> it = topList.iterator();
		for (int i = 0; i < topList.size(); i++) 
			result[i] = it.next().getLabel();
		return result;
	}
	


	@Override
	public int[] explore(int entryLabel, int k, int maxDistanceCalcCount, int[] forbiddenIds) {
		final TreeSet<ObjectDistance> topList = internalGraph.explore(entryLabel, k, maxDistanceCalcCount, forbiddenIds);
		
		final int[] result = new int[topList.size()];
		final Iterator<ObjectDistance> it = topList.iterator();
		for (int i = 0; i < topList.size(); i++) 
			result[i] = it.next().getLabel();
		return result;
	}
	
	
	@Override
	public DynamicExplorationGraph copy() {
		return new DynamicExplorationGraph(internalGraph.copy());
	}

	@Override
	public String toString() {
		return "Size: "+this.size()+", edgesPerVertex: "+this.internalGraph.getEdgesPerVertex();
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
	 * Read meta data for the {@link FeatureSpace}, number of vertices and edges
	 * 
	 * @param file
	 * @return
	 * @throws ClassNotFoundException
	 * @throws IOException
	 */
	public static DynamicExplorationGraph readFromFile(Path file) throws IOException {
		return new DynamicExplorationGraph(MapBasedWeightedUndirectedRegularGraph.readFromFile(file));
	}
	
	/**
	 * Read meta data for the {@link FeatureSpace}, number of vertices and edges
	 * 
	 * @param file
	 * @param featureType
	 * @return
	 * @throws IOException
	 */
	public static DynamicExplorationGraph readFromFile(Path file, String featureType) throws IOException {
		return new DynamicExplorationGraph(MapBasedWeightedUndirectedRegularGraph.readFromFile(file, featureType));
	}
}