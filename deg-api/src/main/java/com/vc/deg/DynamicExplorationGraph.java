package com.vc.deg;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collection;
import java.util.Random;
import java.util.function.IntConsumer;

public interface DynamicExplorationGraph {
	
	/**
	 * Designer to change the graph
	 * 
	 * @return
	 */
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
	 * Create a copy of the graph
	 * 
	 * @return
	 */
	public DynamicExplorationGraph copy();
	
	
	
	
	
	
	/**
	 * Search the graph for the best vertices matching the query.
	 * Find k good vertices and keep searching if any of there
	 * neighbors is better than the worst vertex in the result list. 
	 * 
	 * @param query
	 * @param k
	 * @return
	 */
	public default int[] search(FeatureVector query, int k) {
		return search(Arrays.asList(query), k, 0);
	}
	
	/**
	 * Search the graph for the best vertices matching one of the queries.
	 * The distance to all queries is calculated but only the shortest is kept.
	 * Find k good vertices and keep searching if any of there
	 * neighbors is better than the worst vertex in the result list. 
	 * 
	 * @param queries
	 * @param k
	 * @return
	 */
	public default int[] search(Collection<FeatureVector> queries, int k) {
		return search(queries, k, 0);
	}
	
	/**
	 * Search the graph for the best vertices matching the query.
	 * Find k good vertices and keep searching if any of there
	 * neighbors is better than the worst vertex in the result list
	 * or close to it (search radius: eps) 
	 * 
	 * @param query
	 * @param k
	 * @param eps factor expands the search radius based on the distance to the query. 0 disables the factor, 1 doubles the search radius
	 * @return
	 */
	public default int[] search(FeatureVector query, int k, float eps) {
		return search(Arrays.asList(query), k, eps);
	}
	
	/**
	 * Search the graph for the best vertices matching one of the queries.
	 * The distance to all queries is calculated but only the shortest is kept.
	 * Find k good vertices and keep searching if any of there
	 * neighbors is better than the worst vertex in the result list
	 * or close to it (search radius: eps) 
	 * 
	 * @param queries
	 * @param k
	 * @param eps factor expands the search radius based on the distance to the query. 0 disables the factor, 1 doubles the search radius
	 * @return
	 */
	public default int[] search(Collection<FeatureVector> queries, int k, float eps) {
		return search(queries, k, eps, null);
	}
	
	/**
	 * Search the graph for the best vertices matching the query.
	 * 
	 * Find k good vertices and keep searching if any of there
	 * neighbors is better than the worst vertex in the result list
	 * or close to it (search radius: eps) 
	 * 
	 * Any entry in the returning result list must pass the filter.
	 * 
	 * @param query
	 * @param k
	 * @param eps factor expands the search radius based on the distance to the query. 0 disables the factor, 1 doubles the search radius
	 * @param filter null will be ignored
	 * @return
	 */
	public default int[] search(FeatureVector query, int k, float eps, GraphFilter filter) {
		return search(Arrays.asList(query), k, eps, filter);
	}
	
	/**
	 * Search the graph for the best vertices matching one of the queries.
	 * The distance to all queries is calculated but only the shortest is kept.
	 * 
	 * Find k good vertices and keep searching if any of there
	 * neighbors is better than the worst vertex in the result list
	 * or close to it (search radius: eps) 
	 * 
	 * Any entry in the returning result list must pass the filter.
	 * 
	 * @param queries
	 * @param k
	 * @param eps factor expands the search radius based on the distance to the query. 0 disables the factor, 1 doubles the search radius
	 * @param filter null disables the filter
	 * @return
	 */
	public int[] search(Collection<FeatureVector> queries, int k, float eps, GraphFilter filter);
	
	
	

	
	
	/**
	 * Start from the entry vertex and explore the neighborhood to find k-similar neighbors.
	 * 
	 * Find k good vertices and keep searching if any of there
	 * neighbors is better than the worst vertex in the result list. 
	 * 
	 * @param entryLabel
	 * @param k
	 * @return
	 */
	public default int[] explore(int entryLabel, int k) {
		return explore(new int[] { entryLabel }, k);
	}
	
	/**
	 * Start from the entry vertices and explore the neighborhood to find k-similar neighbors.
	 * The best vertices matching one of the entry vertices are collected.
	 * The distance to all entry vertices is calculated but only the shortest is kept.
	 * 
	 * Find k good vertices and keep searching if any of there
	 * neighbors is better than the worst vertex in the result list. 
	 * 
	 * @param entryLabels
	 * @param k
	 * @return
	 */
	public default int[] explore(int[] entryLabels, int k) {
		return explore(entryLabels, k, Integer.MAX_VALUE);
	}
		
	/**
	 * Start from the entry vertex and explore the neighborhood to find k-similar neighbors.
	 * 
	 * Find k good vertices and keep searching if any of there
	 * neighbors is better than the worst vertex in the result list. 
	 * 
	 * Stop searching if the number of checks (distance calculations) will exceed the
	 * max distance computation count.
	 * 
	 * @param entryLabel
	 * @param k
	 * @param maxDistanceComputationCount
	 * @return
	 */
	public default int[] explore(int entryLabel, int k, int maxDistanceComputationCount) {
		return explore(new int[] {entryLabel}, k, maxDistanceComputationCount);
	}
	
	/**
	 * Start from the entry vertices and explore the neighborhood to find k-similar neighbors.
	 * The best vertices matching one of the entry vertices are collected.
	 * The distance to all entry vertices is calculated but only the shortest is kept.
	 * 
	 * Find k good vertices and keep searching if any of there
	 * neighbors is better than the worst vertex in the result list. 
	 * 
	 * Stop searching if the number of checks (distance calculations) will exceed the
	 * max distance computation count.
	 * 
	 * @param entryLabel
	 * @param k
	 * @param maxDistanceComputationCount
	 * @return
	 */
	public default int[] explore(int[] entryLabel, int k, int maxDistanceComputationCount) {
		return explore(entryLabel, k, maxDistanceComputationCount, null);
	}
		
	/**
	 * Start from the entry vertex and explore the neighborhood to find k-similar neighbors.
	 * 
	 * Find k good vertices and keep searching if any of there
	 * neighbors is better than the worst vertex in the result list. 
	 * 
	 * Stop searching if the number of checks (distance calculations) will exceed the
	 * max distance computation count.
	 * 
	 * Any entry in the returning result list must pass the filter.
	 * 
	 * @param entryLabel
	 * @param k
	 * @param maxDistanceComputationCount
	 * @param filter null disables the filter
	 * @return
	 */
	public default int[] explore(int entryLabel, int k, int maxDistanceComputationCount, GraphFilter filter) {
		return explore(new int[] { entryLabel }, k, maxDistanceComputationCount, filter);
	}
	
	/**
	 * Start from the entry vertices and explore the neighborhood to find k-similar neighbors.
	 * The best vertices matching one of the entry vertices are collected.
	 * The distance to all entry vertices is calculated but only the shortest is kept.
	 * 
	 * Find k good vertices and keep searching if any of there
	 * neighbors is better than the worst vertex in the result list. 
	 * 
	 * Stop searching if the number of checks (distance calculations) will exceed the
	 * max distance computation count.
	 * 
	 * Any entry in the returning result list must pass the filter.
	 * 
	 * @param entryLabel
	 * @param k
	 * @param maxDistanceComputationCount
	 * @param filter null disables the filter
	 * @return
	 */
	public int[] explore(int[] entryLabel, int k, int maxDistanceComputationCount, GraphFilter filter);
	
	/**
	 * Does the graph has a vertex with this label
	 * 
	 * @param label
	 * @return
	 */
	public boolean hasLabel(int label);
	
	/**
	 * The feature space in which the feature vectors of each node 
	 * are embedded and what is used to determine good neighbors.
	 *   
	 * @return
	 */
	public FeatureSpace getFeatureSpace();
	
	
	/**
	 * The Feature Vector of belonging to the label
	 * 
	 * @param label
	 * @return
	 */
	public FeatureVector getFeature(int label);
	
	/**
	 * The number of vertices in the graph
	 * 
	 * @return
	 */
	public int size();
	
	/**
	 * Iterate over all vertices in the graph
	 * 
	 * @param consumer
	 */
	public void forEachVertex(VertexConsumer consumer);

	/**
	 * Iterate over all neighbors of a vertex (given by the label) and consume their ids
	 * 
	 * @param label
	 * @param idConsumer
	 */
	public void forEachNeighbor(int label, IntConsumer idConsumer);
	
	/**
	 * Get a random label
	 * 
	 * @param random
	 * @return
	 */
	public int getRandomLabel(Random random);
	
	
	
	
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
	
	/**
	 * Load an existing graph
	 * 
	 * @param file
	 * @return
	 */
	public static DynamicExplorationGraph loadGraph(Path file, String componentType) throws ClassNotFoundException, IOException {
		return GraphFactory.getDefaultFactory().loadGraph(file, componentType);
	}
}