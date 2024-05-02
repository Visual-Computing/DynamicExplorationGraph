package com.vc.deg.graph;

import java.util.ServiceLoader;
import java.util.function.Consumer;
import java.util.function.IntConsumer;



/**
 * Factory to build graph filter.
 * 
 * Different implementation of the graph filter factory can be registered via a ServiceLoader.
 * Depending on the runtime implementation the {@link #getDefaultFactory()} presents more
 * readable or more efficient graph filter implementations.
 *  * 
 * @author Nico Hezel
 */
public interface VertexFilterFactory {
	
    public static class RegisteredFactoryHolder {
        private static final VertexFilterFactory defaultFactory = ServiceLoader.load(VertexFilterFactory.class).iterator().next();
    }

    /**
     * Returns the default {@link VertexFilterFactory} implementation
     * 
     * @return the default {@link VertexFilterFactory} implementation
     */
    public static VertexFilterFactory getDefaultFactory() {
        return RegisteredFactoryHolder.defaultFactory;
    }
    
    /**
     * Create a graph filter based on the given ids
     * 
     * @param validIds
     * @param allElementCount
     * @return
     */
	public VertexFilter of(int[] validIds, int allElementCount);
	
	/**
	 * Create a graph filter based on the given ids
	 * 
	 * @param validIds
	 * @param allElementCount
	 * @return
	 */
	public VertexFilter of(Consumer<IntConsumer> validIds, int allElementCount);
	
	
	/**
	 * AND operation.
	 * x1 get modified.
	 * 
	 * @param x1
	 * @param x2
	 */
	public void and(VertexFilter x1, VertexFilter x2);
	
	/**
	 * ANDNOT operation. Is used to clear remove all the labels from x1 which are specified in x2.
	 * x1 get modified.
	 * 
	 * @param x1
	 * @param x2
	 */
	public void andNot(VertexFilter x1, VertexFilter x2);
	
	/**
	 * Add elements.
	 * x1 get modified.
	 * 
	 * @param x1
	 * @param x2
	 */
	public void add(VertexFilter x1, Consumer<IntConsumer> x2);
	
	/**
	 * Remove elements.
	 * x1 get modified.
	 * 
	 * @param x1
	 * @param x2
	 */
	public void remove(VertexFilter x1, Consumer<IntConsumer> x2);
}