package com.vc.deg;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collection;
import java.util.Random;
import java.util.function.IntConsumer;

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
	 * Create a copy of the graph
	 * 
	 * @return
	 */
	@Override
	public HierarchicalDynamicExplorationGraph copy();
	
	@Override
	public default int[] search(Collection<FeatureVector> queries, int k, float eps, GraphFilter filter) {
		return searchAtLevel(queries, 0, k, eps, filter);
	}
	
	/**
	 * Get the Dynamic Exploration Graph at the level
	 * 
	 * @param atLevel
	 * @return
	 */
	public DynamicExplorationGraph getGraph(int atLevel);
	
	/**
	 * Search the graph for the best vertices matching the query at the given hierarchy level.
	 * Find k good vertices and keep searching if any of there
	 * neighbors is better than the worst vertex in the result list. 
	 * 
	 * @param query
	 * @param atLevel hierarchy level to search
	 * @param k
	 * @return
	 */
	public default int[] searchAtLevel(FeatureVector query, int atLevel, int k) {
		return searchAtLevel(Arrays.asList(query), atLevel, k, 0);
	}
	
	/**
	 * Search the graph for the best vertices matching one of the queries at the given hierarchy level.
	 * The distance to all queries is calculated but only the shortest is kept.
	 * 
	 * Find k good vertices and keep searching if any of there
	 * neighbors is better than the worst vertex in the result list. 
	 * 
	 * @param queries
	 * @param k
	 * @return
	 */
	public default int[] searchAtLevel(Collection<FeatureVector> queries, int atLevel, int k) {
		return searchAtLevel(queries, atLevel, k, 0);
	}
	
	/**
	 * Search the graph for the best vertices matching the query at the given hierarchy level.
	 * 
	 * Find k good vertices and keep searching if any of there
	 * neighbors is better than the worst vertex in the result list
	 * or close to it (search radius: eps) 
	 * 
	 * @param query
	 * @param atLevel hierarchy level to search
	 * @param k
	 * @param eps factor expands the search radius based on the distance to the query. 0 disables the factor, 1 doubles the search radius
	 * @return
	 */
	public default int[] searchAtLevel(FeatureVector query, int atLevel, int k, float eps) {
		return searchAtLevel(Arrays.asList(query), atLevel, k, eps);
	}
	
	/**
	 * Search the graph for the best vertices matching one of the queries at the given hierarchy level.
	 * The distance to all queries is calculated but only the shortest is kept.
	 * 
	 * Find k good vertices and keep searching if any of there
	 * neighbors is better than the worst vertex in the result list
	 * or close to it (search radius: eps) 
	 * 
	 * @param queries
	 * @param atLevel hierarchy level to search
	 * @param k
	 * @param eps factor expands the search radius based on the distance to the query. 0 disables the factor, 1 doubles the search radius
	 * @return
	 */
	public default int[] searchAtLevel(Collection<FeatureVector> queries, int atLevel, int k, float eps) {
		return searchAtLevel(queries, atLevel, k, eps, null);
	}
	
	/**
	 * Search the graph for the best vertices matching the query at the given hierarchy level.
	 * 
	 * Find k good vertices and keep searching if any of there
	 * neighbors is better than the worst vertex in the result list
	 * or close to it (search radius: eps) 
	 * 
	 * Any entry in the returning result list must pass the filter.
	 * 
	 * @param query
	 * @param atLevel hierarchy level to search
	 * @param k
	 * @param eps factor expands the search radius based on the distance to the query. 0 disables the factor, 1 doubles the search radius
	 * @param filter null will be ignored
	 * @return
	 */
	public default int[] searchAtLevel(FeatureVector query, int atLevel, int k, float eps, GraphFilter filter) {
		return searchAtLevel(Arrays.asList(query), atLevel, k, eps, filter);
	}
	
	/**
	 * Search the graph for the best vertices matching one of the queries at the given hierarchy level.
	 * The distance to all queries is calculated but only the shortest is kept.
	 * 
	 * Find k good vertices and keep searching if any of there
	 * neighbors is better than the worst vertex in the result list
	 * or close to it (search radius: eps) 
	 * 
	 * Any entry in the returning result list must pass the filter.
	 * 
	 * @param queries
	 * @param atLevel hierarchy level to search
	 * @param k
	 * @param eps factor expands the search radius based on the distance to the query. 0 disables the factor, 1 doubles the search radius
	 * @param filter null disables the filter
	 * @return
	 */
	public int[] searchAtLevel(Collection<FeatureVector> queries, int atLevel, int k, float eps, GraphFilter filter);
	
	
	
	
	

	
	
		
	@Override
	public default int[] explore(int[] entryLabel, int k, int maxDistanceComputationCount, GraphFilter filter) {
		return exploreAtLevel(entryLabel, 0, k, maxDistanceComputationCount, filter);
	}
	

	/**
	 * Start from the entry vertex at the given hierarchical layer and explore the neighborhood to find k-similar neighbors.
	 * 
	 * Find k good vertices and keep searching if any of there
	 * neighbors is better than the worst vertex in the result list. 
	 * 
	 * @param entryLabel
	 * @param atLevel hierarchy level to search
	 * @param k
	 * @return
	 */
	public default int[] exploreAtLevel(int entryLabel, int atLevel, int k) {
		return exploreAtLevel(new int[] { entryLabel }, atLevel, k);
	}
	
	/**
	 * Start from the entry vertices at the given hierarchical layer and explore the neighborhood to find k-similar neighbors.
	 * The best vertices matching one of the entry vertices are collected.
	 * The distance to all entry vertices is calculated but only the shortest is kept.
	 * 
	 * Find k good vertices and keep searching if any of there
	 * neighbors is better than the worst vertex in the result list. 
	 * 
	 * @param entryLabels
	 * @param atLevel hierarchy level to search
	 * @param k
	 * @return
	 */
	public default int[] exploreAtLevel(int[] entryLabels, int atLevel, int k) {
		return exploreAtLevel(entryLabels, atLevel, k, Integer.MAX_VALUE);
	}
		
	/**
	 * Start from the entry vertex at the given hierarchical layer and explore the neighborhood to find k-similar neighbors.
	 * 
	 * Find k good vertices and keep searching if any of there
	 * neighbors is better than the worst vertex in the result list. 
	 * 
	 * Stop searching if the number of checks (distance calculations) will exceed the
	 * max distance computation count.
	 * 
	 * @param entryLabel
	 * @param atLevel hierarchy level to search
	 * @param k
	 * @param maxDistanceComputationCount
	 * @return
	 */
	public default int[] exploreAtLevel(int entryLabel, int atLevel, int k, int maxDistanceComputationCount) {
		return exploreAtLevel(new int[] {entryLabel}, atLevel, k, maxDistanceComputationCount);
	}
	
	/**
	 * Start from the entry vertices at the given hierarchical layer and explore the neighborhood to find k-similar neighbors.
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
	 * @param atLevel hierarchy level to search
	 * @param k
	 * @param maxDistanceComputationCount
	 * @return
	 */
	public default int[] exploreAtLevel(int[] entryLabel, int atLevel, int k, int maxDistanceComputationCount) {
		return exploreAtLevel(entryLabel, atLevel, k, maxDistanceComputationCount, null);
	}
		
	/**
	 * Start from the entry vertex at the given hierarchical layer and explore the neighborhood to find k-similar neighbors.
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
	 * @param atLevel hierarchy level to search
	 * @param k
	 * @param maxDistanceComputationCount
	 * @param filter null disables the filter
	 * @return
	 */
	public default int[] exploreAtLevel(int entryLabel, int atLevel, int k, int maxDistanceComputationCount, GraphFilter filter) {
		return exploreAtLevel(new int[] { entryLabel }, atLevel, k, maxDistanceComputationCount, filter);
	}
	
	/**
	 * Start from the entry vertices at the given hierarchical layer and explore the neighborhood to find k-similar neighbors.
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
	 * @param atLevel hierarchy level to search
	 * @param k
	 * @param maxDistanceComputationCount
	 * @param filter null disables the filter
	 * @return
	 */
	public int[] exploreAtLevel(int[] entryLabel, int atLevel, int k, int maxDistanceComputationCount, GraphFilter filter);
	
	/**
	 * Size of the graph at the given level
	 * 
	 * @param atLevel
	 * @return
	 */
	public int sizeAtLevel(int atLevel);
		
	@Override
	default int size() {
		return sizeAtLevel(0);
	}
	
	/**
	 * Iterate over all vertices in the graph
	 * 
	 * @param atLevel
	 * @param consumer
	 */
	public void forEachVertexAtLevel(int atLevel, VertexConsumer consumer);
	
	@Override
	default void forEachVertex(VertexConsumer consumer) {
		forEachVertexAtLevel(0, consumer);
	}
	
	/**
	 * Iterate over all neighbors of a vertex at a specific level and consume their ids
	 * 
	 * @param atLevel
	 * @param label
	 * @param idConsumer
	 */
	public void forEachNeighborAtLevel(int atLevel, int label, IntConsumer idConsumer);
	
	@Override
	default void forEachNeighbor(int label, IntConsumer idConsumer) {
		forEachNeighborAtLevel(0, label, idConsumer);
	}
	
	/**
	 * Does a vertex with the given label exists on the level
	 * 
	 * @param label
	 * @param atLevel
	 * @return
	 */
	public boolean hasLabelAtLevel(int label, int atLevel);
	
	@Override
	default boolean hasLabel(int label) {
		return hasLabelAtLevel(label, 0);
	}
	
	/**
	 * Get a random label at the level
	 * 
	 * @param random
	 * @param atLevel
	 * @return
	 */
	public int getRandomLabelAtLevel(Random random, int atLevel);
	
	@Override
	default int getRandomLabel(Random random) {
		return getRandomLabelAtLevel(random, 0);
	}
	
	/**
	 * Number of levels of the hierarchical graph
	 * 
	 * @return
	 */
	public int levelCount();
	
	
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
