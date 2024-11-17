package com.vc.deg.ref;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.TreeSet;
import java.util.stream.Collectors;

import com.vc.deg.FeatureSpace;
import com.vc.deg.FeatureVector;
import com.vc.deg.graph.VertexFilter;
import com.vc.deg.graph.NeighborConsumer;
import com.vc.deg.graph.VertexCursor;
import com.vc.deg.ref.designer.EvenRegularGraphDesigner;
import com.vc.deg.ref.graph.ArrayBasedWeightedUndirectedRegularGraph;
import com.vc.deg.ref.graph.QueryDistance;
import com.vc.deg.ref.graph.VertexData;


/**
 * This class wraps the main functions of the library
 * 
 * @author Nico Hezel
 */
public class DynamicExplorationGraph implements com.vc.deg.DynamicExplorationGraph {

	protected final ArrayBasedWeightedUndirectedRegularGraph internalGraph;
	protected final EvenRegularGraphDesigner designer;

	/**
	 * Define how the vertices of the graph will be compared and how many edges each vertex has.
	 * 
	 * @param space
	 * @param edgesPerVertex
	 */
	public DynamicExplorationGraph(FeatureSpace space, int edgesPerVertex) {
		this.internalGraph = new ArrayBasedWeightedUndirectedRegularGraph(edgesPerVertex, space);
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
		this.internalGraph = new ArrayBasedWeightedUndirectedRegularGraph(edgesPerVertex, expectedSize, space);
		this.designer = new EvenRegularGraphDesigner(internalGraph); 
	}
	
	/**
	 * Used by the copy constructor and read file methods
	 * 
	 * @param graph
	 */
	private DynamicExplorationGraph(ArrayBasedWeightedUndirectedRegularGraph graph) {
		this.internalGraph = graph;
		this.designer = new EvenRegularGraphDesigner(internalGraph); 
	}
	
	/**
	 * Any changes to the graph might destroy the Dynamic Exploration Graph properties.
	 * 
	 * @return
	 */
	public ArrayBasedWeightedUndirectedRegularGraph getInternalGraph() {
		return internalGraph;
	}
	
	@Override
	public VertexCursor vertexCursor() {
		return new com.vc.deg.ref.graph.VertexCursor(this.internalGraph);
	}
	
	@Override
	public VertexFilter labelFilter() {
		return this.internalGraph.labelsFilter();
	}
	
	@Override
	public void forEachNeighbor(int label, NeighborConsumer neighborConsumer) {
		this.internalGraph.getVertexByLabel(label).getEdges().forEach((int neighborId, float weight) -> 
			neighborConsumer.accept(this.internalGraph.getVertexById(neighborId).getLabel(), weight)
		);
	}
	
	@Override
	public int getRandomLabel(Random random) {
		return this.internalGraph.getRandomVertex(random).getLabel();
	}
	
	@Override
	public int getRandomLabel(Random random, VertexFilter filter) {
		final VertexData vertex = this.internalGraph.getRandomVertex(random, filter);
		return (vertex == null) ? -1 : vertex.getLabel();
	}
	
	@Override
	public FeatureVector getFeature(int label) {
		return this.internalGraph.getVertexByLabel(label).getFeature();
	}

	@Override
	public boolean hasLabel(int label) {
		return this.internalGraph.hasVertex(label);
	}

	@Override
	public FeatureSpace getFeatureSpace() {
		return this.internalGraph.getFeatureSpace();
	}
	
	@Override
	public int size() {
		return this.internalGraph.getVertexCount();
	}
	
	@Override
	public int edgePerVertex() {
		return this.internalGraph.getEdgesPerVertex();
	}
	
	@Override
	public EvenRegularGraphDesigner designer() {
		return designer;
	}
	
	@Override
	public int[] search(Collection<FeatureVector> queries, int k, float eps, VertexFilter filter, int[] seedVertexLabels) {
		int[] seedVertexIds = Arrays.stream(seedVertexLabels).mapToObj(internalGraph::getVertexByLabel).filter(Objects::nonNull).mapToInt(VertexData::getId).toArray();
		if(seedVertexIds.length == 0)
			seedVertexIds = new int[] { internalGraph.getVertices().iterator().next().getId() };
		
		final TreeSet<QueryDistance> topList = internalGraph.search(queries, k, eps, filter, seedVertexIds, true);
		final int[] result = new int[topList.size()];
		final Iterator<QueryDistance> it = topList.iterator();
		for (int i = 0; i < topList.size(); i++) 
			result[i] = it.next().getVertexLabel();
		return result;
	}

	@Override
	public int[] explore(int[] seedVertexLabels, int k, float eps, VertexFilter filter) {
		
		// check seed labels
		final VertexData[] entrys = Arrays.stream(seedVertexLabels).mapToObj(internalGraph::getVertexByLabel).filter(Objects::nonNull).toArray(VertexData[]::new);
		if(entrys.length == 0)
			throw new RuntimeException("None of the seed labels "+Arrays.toString(seedVertexLabels)+" are in the graph.");
		
		// prepare search
		final int[] entryIds = Arrays.stream(entrys).mapToInt(VertexData::getId).toArray();
		final List<FeatureVector> queries = Arrays.stream(entrys).map(VertexData::getFeature).collect(Collectors.toList());		
		final TreeSet<QueryDistance> topList = internalGraph.search(queries, k, eps, filter, entryIds, false);
		
		// convert to int array
		final int[] result = new int[topList.size()];
		final Iterator<QueryDistance> it = topList.iterator();
		for (int i = 0; i < topList.size(); i++) 
			result[i] = it.next().getVertexLabel();
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
	 * Goal: Write should be compatible with deglib in c-lang. 
	 * 		- all internal id are between 0 to n where n is the number of vertices (no holes)
	 * 		- the graph is stored in little endian format
	 * 		-
	 * 
	 * 
	 * 
	 * Problem: is the endians and the FeatureVector object type written into the output file.
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
	 * @throws IOException
	 */
	public static DynamicExplorationGraph readFromFile(Path file) throws IOException {
		return new DynamicExplorationGraph(ArrayBasedWeightedUndirectedRegularGraph.readFromFile(file));
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
		return new DynamicExplorationGraph(ArrayBasedWeightedUndirectedRegularGraph.readFromFile(file, featureType));
	}
}