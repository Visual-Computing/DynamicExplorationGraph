package com.vc.deg.viz.feature;

import java.util.HashMap;
import java.util.Map;

import com.vc.deg.FeatureFactory;
import com.vc.deg.FeatureSpace;
import com.vc.deg.FeatureVector;
import com.vc.deg.feature.BinaryFeature;

public interface FeatureTransformer {
	

	/**
	 * Contains all {@link FeatureFactory}s registered manually
	 *  
	 * @author Nico Hezel
	 */
    public static class TransformerFactory {

    	private final static Map<FeatureSpace, FeatureTransformer> registeredTransformers = new HashMap<>();
    	    	
    	/**
    	 * Register a new {@link FeatureFactory} manual
    	 * 
    	 * @param factory
    	 * @param space
    	 */
    	private static void registerTransformer(FeatureTransformer factory, FeatureSpace space) {
    		registeredTransformers.put(space, factory);
    	}
    	
    	/**
    	 * Find a specific {@link TransformerFactory} based on the parameters
    	 * 
    	 * @param featureSpace
    	 * @return
    	 */
    	private static FeatureTransformer findTransformer(FeatureSpace featureSpace) {
			return registeredTransformers.getOrDefault(featureSpace, null);
    	}
    }

    /**
	 * Register a new {@link FeatureTransformer} for a given {@link FeatureSpace}
	 * 
     * @param factory
     * @param space
     */
    public static void registerTransformer(FeatureTransformer factory, FeatureSpace space) {
    	TransformerFactory.registerTransformer(factory, space);
    }
    
    /**
	 * Find a specific {@link FeatureTransformer} based on the parameters
	 * 
     * @param space
     * @return
     */
    public static FeatureTransformer findTransformer(FeatureSpace space) {
    	final FeatureTransformer transformer = TransformerFactory.findTransformer(space);
    	if(transformer != null)
    		return transformer;
    	else if(byte.class == space.getComponentType())
			return new ByteToFloatTransformer(space.dims());
		else if(BinaryFeature.ComponentType == space.getComponentType())
			return new BinaryToFloatTransformer(space.dims());

		throw new RuntimeException("Could not find float trnasformer for feature space "+space.getComponentType()+" with "+space.dims());
    }
    
    
	/**
	 * Output dimensions
	 * @return
	 */
	public int dims();

	/**
	 * Transformed feature
	 * 
	 * @param fv
	 * @return
	 */
	public float[] transform(FeatureVector fv);
	
	
	
	public static class BinaryToFloatTransformer implements FeatureTransformer {
				
		public static final int StepSize = 1;
		protected final int dims;
		
		public BinaryToFloatTransformer(int dims) {
			this.dims = dims;
		}
		
		@Override
		public int dims() {
			return dims / StepSize;
		}

		@Override
		public float[] transform(FeatureVector fv) {
			
			final int bits = 64;
			final int longCount = dims / bits;
			final int reducedDims = dims / StepSize;
			final float[] result = new float[reducedDims];
			for (int l = 0, d = 0, b = 0; l < longCount; l++) {
				final long input = fv.readLong(l * Long.BYTES);
				for(; b < bits; b+=StepSize) 
					result[d++] = (input >> b & 1);
				b -= bits;
			}			
			
			return result;
		}		
	}

	
	
	public static class ByteToFloatTransformer implements FeatureTransformer {
		
		protected final int dims;
		
		public ByteToFloatTransformer(int dims) {
			this.dims = dims;
		}
		
		@Override
		public int dims() {
			return dims;
		}

		@Override
		public float[] transform(FeatureVector fv) {
			final float[] result = new float[dims];
			for (int i = 0; i < result.length; i++) 
				result[i] = fv.readByte(i * Byte.BYTES);
			return result;
		}		
	}
}