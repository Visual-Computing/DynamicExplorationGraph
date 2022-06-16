package com.vc.deg;

import java.nio.file.Path;
import java.util.Iterator;
import java.util.ServiceLoader;



/**
 * 
 * use ServiceLoader to register Dynamic Exploration Graphs
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
	 * Load an existing graph
	 * 
	 * @param file
	 * @return
	 */
	public DynamicExplorationGraph loadGraph(Path file);
}
