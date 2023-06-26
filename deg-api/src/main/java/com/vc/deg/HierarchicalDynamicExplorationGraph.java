package com.vc.deg;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collection;
import java.util.Random;

import com.vc.deg.graph.GraphDesigner;
import com.vc.deg.graph.GraphFilter;
import com.vc.deg.graph.NeighborConsumer;
import com.vc.deg.graph.VertexCursor;

public interface HierarchicalDynamicExplorationGraph extends DynamicExplorationGraph {

	/**
	 * Designer to change the graph
	 * 
	 * @return
	 */
	@Override
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
	
	/**
	 * Get the Dynamic Exploration Graph at the level
	 * 
	 * @param atLevel
	 * @return
	 */
	public DynamicExplorationGraph getGraph(int atLevel);
	
	@Override
	public default int[] search(Collection<FeatureVector> queries, int k, float eps, GraphFilter filter, int[] seedVertexLabels) {
		return searchAtLevel(queries, 0, k, eps, filter, seedVertexLabels);
	}
	
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
	public default int[] searchAtLevel(Collection<FeatureVector> queries, int atLevel, int k, float eps, GraphFilter filter) {
		return searchAtLevel(queries, atLevel, k, eps, filter, new int[0]);
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
	 * The search starts at the seedVertexLabels.
	 * 
	 * @param queries
	 * @param atLevel hierarchy level to search
	 * @param k
	 * @param eps factor expands the search radius based on the distance to the query. 0 disables the factor, 1 doubles the search radius
	 * @param filter null disables the filter
	 * @param seedVertexLabels if empty or filled with invalid ids, the default starting point will be used instead
	 * @return
	 */
	public int[] searchAtLevel(Collection<FeatureVector> queries, int atLevel, int k, float eps, GraphFilter filter, int[] seedVertexLabels);
	
	
	
	

	
	
		
	@Override
	public default int[] explore(int[] seedLabel, int k, float eps, GraphFilter filter) {
		return exploreAtLevel(seedLabel, 0, k, eps, filter);
	}
	

	/**
	 * Start from the entry vertex at the given hierarchical layer and explore the neighborhood to find k-similar neighbors.
	 * 
	 * Find k good vertices and keep searching if any of there
	 * neighbors is better than the worst vertex in the result list. 
	 * 
	 * Any entry in the returning result is not the seed label.
	 * 
	 * @param seedLabel
	 * @param atLevel hierarchy level to search
	 * @param k
	 * @return
	 */
	public default int[] exploreAtLevel(int seedLabel, int atLevel, int k) {
		return exploreAtLevel(new int[] { seedLabel }, atLevel, k);
	}
	
	/**
	 * Start from the entry vertices at the given hierarchical layer and explore the neighborhood to find k-similar neighbors.
	 * The best vertices matching one of the entry vertices are collected.
	 * The distance to all entry vertices is calculated but only the shortest is kept.
	 * 
	 * Find k good vertices and keep searching if any of there
	 * neighbors is better than the worst vertex in the result list. 
	 * 
	 * Any entry in the returning result list is not in the list of seed labels.
	 * 
	 * @param seedLabels
	 * @param atLevel hierarchy level to search
	 * @param k
	 * @return
	 */
	public default int[] exploreAtLevel(int[] seedLabels, int atLevel, int k) {
		return exploreAtLevel(seedLabels, atLevel, k, Integer.MAX_VALUE);
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
	 * Any entry in the returning result is not the seed label.
	 * 
	 * @param seedLabel
	 * @param atLevel hierarchy level to search
	 * @param k
	 * @param eps
	 * @return
	 */
	public default int[] exploreAtLevel(int seedLabel, int atLevel, int k, float eps) {
		return exploreAtLevel(new int[] {seedLabel}, atLevel, k, eps);
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
	 * Any entry in the returning result list is not in the list of seed labels.
	 * 
	 * @param seedLabels
	 * @param atLevel hierarchy level to search
	 * @param k
	 * @param eps
	 * @return
	 */
	public default int[] exploreAtLevel(int[] seedLabels, int atLevel, int k, float eps) {
		return exploreAtLevel(seedLabels, atLevel, k, eps, null);
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
	 * Any entry in the returning result list must pass the filter and is not the seed label.
	 * 
	 * @param seedLabel
	 * @param atLevel hierarchy level to search
	 * @param k
	 * @param eps
	 * @param filter null disables the filter
	 * @return
	 */
	public default int[] exploreAtLevel(int seedLabel, int atLevel, int k, float eps, GraphFilter filter) {
		return exploreAtLevel(new int[] { seedLabel }, atLevel, k, eps, filter);
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
	 * Any entry in the returning result list must pass the filter and is not in the list of seed labels.
	 * 
	 * @param seedLabels
	 * @param atLevel hierarchy level to search
	 * @param k
	 * @param eps
	 * @param filter null disables the filter
	 * @return
	 */
	public int[] exploreAtLevel(int[] seedLabels, int atLevel, int k, float eps, GraphFilter filter);
	
	
	
	
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
	 * Iterate over all vertices at a specific level
	 * 
	 * @param atLevel
	 */
	public VertexCursor vertexCursorAtLevel(int atLevel);
	
	@Override
	default VertexCursor vertexCursor() {
		return vertexCursorAtLevel(0);
	}
	
	/**
	 * Iterate over all neighbors of a vertex at a specific level and consume their ids
	 * 
	 * @param atLevel
	 * @param label
	 * @param neighborConsumer
	 */
	public void forEachNeighborAtLevel(int atLevel, int label, NeighborConsumer neighborConsumer);
	
	@Override
	default void forEachNeighbor(int label, NeighborConsumer neighborConsumer) {
		forEachNeighborAtLevel(0, label, neighborConsumer);
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
	 * Get a random label at the level which is allowed by the filter
	 * 
	 * @param random
	 * @param atLevel
	 * @param filter
	 * @return
	 */
	public int getRandomLabelAtLevel(Random random, int atLevel, GraphFilter filter);
	
	@Override
	default int getRandomLabel(Random random, GraphFilter filter) {
		return getRandomLabelAtLevel(random, 0, filter);
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
	public static HierarchicalDynamicExplorationGraph newGraph(FeatureSpace space, int edgesPerVertex, int topRankSize) {
		return GraphFactory.getDefaultFactory().newHierchicalGraph(space, edgesPerVertex, topRankSize);
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
