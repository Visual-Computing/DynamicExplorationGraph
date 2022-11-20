package com.vc.deg;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ServiceLoader;


/**
 * Factory to build or load graphs.
 * 
 * Different implementation of the graph factory can be registered via a ServiceLoader.
 * Depending on the runtime implementation the {@link #getDefaultFactory()} presents more
 * readable or more efficient graph implementations.
 * 
 * https://docs.oracle.com/javase/8/docs/api/java/util/ServiceLoader.html
 * https://riptutorial.com/java/example/19523/simple-serviceloader-example
 * 
 * @author Nico Hezel
 */
public interface GraphFactory {
	
    public static class RegisteredFactoryHolder {
        private static final GraphFactory defaultFactory = ServiceLoader.load(GraphFactory.class).iterator().next();
    }

    /**
     * Returns the default {@link GraphFactory} implementation
     * 
     * @return the default {@link GraphFactory} implementation
     */
    public static GraphFactory getDefaultFactory() {
        return RegisteredFactoryHolder.defaultFactory;
    }
    
    
    
	// --------------------------------------------------------------------------------------
	// -------------------------------- Simple Graph ----------------------------------------
	// --------------------------------------------------------------------------------------
    
    /**
     * Create an empty new graph
     * 
     * @param space
     * @param edgesPerNode
     * @return
     */
	public DynamicExplorationGraph newGraph(FeatureSpace space, int edgesPerNode);
	
	/**
	 * Load an existing graph. Read the feature type from the filename.
	 * e.g. sift1m_k30.float.deg 
	 * 
	 * @param file
	 * @return
	 */
	public DynamicExplorationGraph loadGraph(Path file) throws IOException;
	
	/**
	 * Load an existing graph and expect a specific feature type
	 * 
	 * @param file
	 * @param featureType
	 * @return
	 * @throws IOException
	 */
	public DynamicExplorationGraph loadGraph(Path file, String featureType) throws IOException;
	
	
	
	// --------------------------------------------------------------------------------------
	// ---------------------------- Hierarchical Graph --------------------------------------
	// --------------------------------------------------------------------------------------
	   
    /**
     * Create an empty new graph
     * 
	 * @param space
	 * @param edgesPerNode
	 * @param topRankSize
	 * @return
	 */
	public HierarchicalDynamicExplorationGraph newHierchicalGraph(FeatureSpace space, int edgesPerNode, int topRankSize);
	
	/**
	 * Load an existing graph. Read the feature type from the filename.
	 * e.g. sift1m_k30.float.deg 
	 * 
	 * @param file
	 * @return
	 */
	public HierarchicalDynamicExplorationGraph loadHierchicalGraph(Path file) throws IOException;
}
