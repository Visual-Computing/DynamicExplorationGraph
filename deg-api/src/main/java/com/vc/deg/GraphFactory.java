package com.vc.deg;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ServiceLoader;


/**
 * Use ServiceLoader to register Dynamic Exploration Graphs
 * 
 * TODO move to DynamicExplorationGraph.Factory
 * 
 * https://docs.oracle.com/javase/8/docs/api/java/util/ServiceLoader.html
 * https://riptutorial.com/java/example/19523/simple-serviceloader-example
 * 
 * @author Nico Hezel
 */
public interface GraphFactory {
	
    public static class DefaultFactoryHolder {
        private static final GraphFactory defaultFactory = ServiceLoader.load(GraphFactory.class).iterator().next();
    }

    /**
     * Returns the default {@link HashLongIntMapFactory} implementation, to which
     * all static methods in this class delegate.
     *
     
     
     * @return the default {@link HashLongIntMapFactory} implementation
     * @throws RuntimeException if no implementations
     *         of {@link HashLongIntMapFactory} are provided
     */
    public static GraphFactory getDefaultFactory() {
        return DefaultFactoryHolder.defaultFactory;
    }
    
    /**
     * Create an empty new graph
     * 
     * @param space
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
	 * @param componentType
	 * @return
	 * @throws IOException
	 */
	public DynamicExplorationGraph loadGraph(Path file, String featureType) throws IOException;
}
