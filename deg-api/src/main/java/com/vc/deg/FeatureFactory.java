package com.vc.deg;

import java.io.DataInput;
import java.io.IOException;
import java.util.HashSet;
import java.util.ServiceLoader;
import java.util.Set;

/**
 * A {@link FeatureFactory} is needed when a graph is loaded from a drive and contains custom {@link FeatureVector}.
 * 
 * 
 * TODO rework or remove the following text
 * Can use native of on-heap memory depending on the implementation 
 * Goal: single morph calls of the FeatureVector methods in the computeDistance-method of the FeatureSpace, native support and primitive-arrays for fast on-heap computations.
 * 		 Adding a node or Queries could copy the feature first to native memory if needed. 
 * FeatureFactory in reference-impl should always use column-based arrays when reading from file
 *  
 * @author Nico Hezel
 */
public interface FeatureFactory {
	
	/**
	 * Contains all registered {@link FeatureFactory} either via a service loader or manual
	 * 
	 * @author Nico Hezel
	 */
    public static class DefaultFactoryHolder {

    	
    	private final static Set<FeatureFactory> registeredFactories = serviceLoaderFactories();
    	
    	/**
    	 * Get all factories registered via a service loader
    	 * 
    	 * @return
    	 */
    	private static Set<FeatureFactory> serviceLoaderFactories() {
    		final Set<FeatureFactory> serviceLoaderFactories = new HashSet<>();
    		for (FeatureFactory featureFactory : ServiceLoader.load(FeatureFactory.class))
    			serviceLoaderFactories.add(featureFactory);
    		return serviceLoaderFactories;
    	}
    	
    	/**
    	 * Register a new {@link FeatureFactory} manual
    	 * 
    	 * @param factory
    	 */
    	private static void registerFactory(FeatureFactory factory) {
    		if(registeredFactories.contains(factory) == false)
    			registeredFactories.add(factory);
    	}
    	
    	/**
    	 * Find a specific {@link FeatureFactory} based on the parameters
    	 * 
    	 * @param componentType
    	 * @param dims
    	 * @return
    	 */
    	private static FeatureFactory findFactory(String componentType, int dims) {
    		for (FeatureFactory registeredFactory : registeredFactories) 
    			if(componentType.equalsIgnoreCase(registeredFactory.getComponentType().getSimpleName()) && dims == registeredFactory.dims())
    				return registeredFactory;
			return null;
    	}
    	
    	/**
    	 * Find a specific {@link FeatureFactory} based on the parameters
    	 * 
    	 * @param componentType
    	 * @param dims
    	 * @return
    	 */
    	private static FeatureFactory findFactory(Class<?> componentType, int dims) {
    		for (FeatureFactory registeredFactory : registeredFactories) 
    			if(componentType == registeredFactory.getComponentType() && dims == registeredFactory.dims())
    				return registeredFactory;
			return null;
    	}
    }

    /**
	 * Register a new {@link FeatureFactory} manual
	 * 
	 * @param factory
	 */
    public static void registerFactory(FeatureFactory factory) {
        DefaultFactoryHolder.registerFactory(factory);
    }
    
    /**
	 * Find a specific {@link FeatureFactory} based on the parameters
	 * 
     * @param componentType
     * @param dims
     * @return
     */
    public static FeatureFactory findFactory(String componentType, int dims) {
        return DefaultFactoryHolder.findFactory(componentType, dims);
    }
    
    /**
	 * Find a specific {@link FeatureFactory} based on the parameters
	 * 
     * @param componentType
     * @param dims
     * @return
     */
    public static FeatureFactory findFactory(Class<?> componentType, int dims) {
        return DefaultFactoryHolder.findFactory(componentType, dims);
    }
    
    /**
     * Either one of the primitives or an object
     * 
     * @return
     */
    public Class<?> getComponentType();
    
    /**
     * Size of the feature vectors in bytes
     * 
     * @return
     */
    public int featureSize();
    
    /**
     * Dimensions of the feature
     * 
     * @return
     */
    public int dims();
    
//    /**
//     * TODO should be (ByteOrder.LITTLE_ENDIAN) the method is the counterpart to FeatureVector#toBytes()
//	   * TODO Having toBytes here requires the specification of a ByteOrder. 
//	   *	  The API should be free from such prefefined rules.
//     *	  Use read(...) instead
//     *
//     * Extract the feature from a pure byte array.
//     * There is not length check between the expected dimensions and the length of the given array.
//     * 
//     * @param featureByte
//     * @return
//     */
//    public FeatureVector of(byte[] featureByte);
    
    /**
     * Read the feature from a data input source
     * There is not length check between the expected dimensions and the end of the stream.
     * 
     * @param is
     * @return
     */
    public FeatureVector read(DataInput is) throws IOException;
}