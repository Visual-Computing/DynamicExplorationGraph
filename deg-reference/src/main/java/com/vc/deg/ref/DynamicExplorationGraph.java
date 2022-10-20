package com.vc.deg.ref;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Collection;
import java.util.Iterator;
import java.util.Random;
import java.util.TreeSet;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;

import com.vc.deg.FeatureSpace;
import com.vc.deg.FeatureVector;
import com.vc.deg.graph.GraphFilter;
import com.vc.deg.graph.VertexConsumer;
import com.vc.deg.ref.designer.EvenRegularGraphDesigner;
import com.vc.deg.ref.graph.ArrayBasedWeightedUndirectedRegularGraph;
import com.vc.deg.ref.graph.VertexData;
import com.vc.deg.ref.graph.VertexDistance;


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
	
	
	/**
	 * Iterate all vertices and call the consumer with every vertex label and feature vector
	 * 
	 * @param consumer
	 */
	@Override
	public void forEachVertex(VertexConsumer consumer) {
		this.internalGraph.getVertices().forEach(vertex -> consumer.accept(vertex.getLabel(), vertex.getFeature()));
	}
	
	@Override
	public void forEachNeighbor(int label, IntConsumer idConsumer) {
		this.internalGraph.getVertexByLabel(label).getEdges().keySet().forEach((int neighborId) -> 
			idConsumer.accept(this.internalGraph.getVertexById(neighborId).getLabel())
		);
	}
	
	@Override
	public int getRandomLabel(Random random) {
		final int steps = random.nextInt(size());
		final Iterator<VertexData> it = this.internalGraph.getVertices().iterator();
		for (int i = 0; i < steps; i++) 
			it.next();		
		return it.next().getLabel();
	}
	
	@Override
	public int getRandomLabel(Random random, GraphFilter filter) {
		int label = -1;
		int lowestIndex = size();
		do {

			final int index = random.nextInt(lowestIndex);
			final Iterator<VertexData> it = this.internalGraph.getVertices().iterator();
			for (int i = 0; i < index; i++) 
				it.next();		
			label = it.next().getLabel();
			
			// test the next element if the random label does not pass the filter
			while(filter.isValid(label) == false && it.hasNext())
				label = it.next().getLabel();
		
			// all vertices after the lowestIndex have already been tested
			lowestIndex = Math.min(lowestIndex, index);
			
		// if lowest index reaches 0 we have tested all elements against the filter
		} while(lowestIndex > 0 && filter.isValid(label) == false);
		
		return label;
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
	public EvenRegularGraphDesigner designer() {
		return designer;
	}
	

	@Override
	public int[] search(Collection<FeatureVector> queries, int k, float eps, GraphFilter filter) {
		final int[] entryVertexId = new int[] { internalGraph.getVertices().iterator().next().getId() };
		final TreeSet<VertexDistance> topList = internalGraph.search(queries, k, eps, filter, entryVertexId);
		
		final int[] result = new int[topList.size()];
		final Iterator<VertexDistance> it = topList.iterator();
		for (int i = 0; i < topList.size(); i++) 
			result[i] = it.next().getVertexLabel();
		return result;
	}

	@Override
	public int[] explore(int[] entryLabels, int k, int maxDistanceComputationCount, GraphFilter filter) {
	final int[] entryIds = IntStream.of(entryLabels).map(label -> internalGraph.getVertexByLabel(label).getId()).toArray();
	final TreeSet<VertexDistance> topList = internalGraph.explore(entryIds, k, maxDistanceComputationCount, filter);
		
		final int[] result = new int[topList.size()];
		final Iterator<VertexDistance> it = topList.iterator();
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