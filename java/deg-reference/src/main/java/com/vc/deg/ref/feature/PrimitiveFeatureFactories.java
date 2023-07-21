package com.vc.deg.ref.feature;

import java.io.DataInput;
import java.io.IOException;

import com.vc.deg.FeatureFactory;
import com.vc.deg.FeatureVector;
import com.vc.deg.feature.BinaryFeature;
import com.vc.deg.feature.ByteFeature;
import com.vc.deg.feature.DoubleFeature;
import com.vc.deg.feature.FloatFeature;
import com.vc.deg.feature.IntFeature;
import com.vc.deg.feature.LongFeature;
import com.vc.deg.feature.ShortFeature;

/**
 * Only used when reading graph from file. 
 * 
 * @author Nico Hezel
 *
 */
public class PrimitiveFeatureFactories {
	
	public static FeatureFactory create(String componentType, int dims) {
		if(byte.class.getSimpleName().equalsIgnoreCase(componentType))
			return new ByteFeatureFactory(dims);
		else if(short.class.getSimpleName().equalsIgnoreCase(componentType))
			return new ShortFeatureFactory(dims);
		else if(int.class.getSimpleName().equalsIgnoreCase(componentType))
			return new IntFeatureFactory(dims);
		else if(long.class.getSimpleName().equalsIgnoreCase(componentType))
			return new LongFeatureFactory(dims);
		else if(float.class.getSimpleName().equalsIgnoreCase(componentType))
			return new FloatFeatureFactory(dims);
		else if(double.class.getSimpleName().equalsIgnoreCase(componentType))
			return new DoubleFeatureFactory(dims);
		else if(BinaryFeature.ComponentType.getSimpleName().equalsIgnoreCase(componentType))
			return new BinaryFeatureFactory(dims);
		return null;
	}
	
	public static FeatureFactory create(Class<?> componentType, int dims) {
		if(byte.class == componentType)
			return new ByteFeatureFactory(dims);
		else if(short.class == componentType)
			return new ShortFeatureFactory(dims);
		else if(int.class == componentType)
			return new IntFeatureFactory(dims);
		else if(long.class == componentType)
			return new LongFeatureFactory(dims);
		else if(float.class == componentType)
			return new FloatFeatureFactory(dims);
		else if(double.class == componentType)
			return new DoubleFeatureFactory(dims);
		else if(BinaryFeature.ComponentType == componentType)
			return new BinaryFeatureFactory(dims);
		return null;
	}
	
	/**
	 * Factory for the {@link ByteFeature}
	 * 
	 * @author Nico Hezel
	 */
	public static class BinaryFeatureFactory implements FeatureFactory {
		
		protected final int dims;
		protected final int longCount;
		
		public BinaryFeatureFactory(int dims) {
			this.dims = dims;	
			this.longCount =  (int) Math.ceil((float)dims / 64);
			if(dims > 0xFFFF)
				throw new UnsupportedOperationException("Features with more than "+0xFFFF+" dimensions are not supported");
		}

		@Override
		public Class<?> getComponentType() {
			return BinaryFeature.ComponentType;
		}

		@Override
		public int featureSize() {
			return longCount * Long.BYTES;
		}

		@Override
		public int dims() {
			return dims;
		}

		@Override
		public FeatureVector read(DataInput is) throws IOException {
			final long[] feature = new long[longCount];
			for (int i = 0; i < feature.length; i++) 
				feature[i] = is.readLong();
			return new BinaryFeature(dims, feature);
		}
	}
	
	/**
	 * Factory for the {@link ByteFeature}
	 * 
	 * @author Nico Hezel
	 */
	public static class ByteFeatureFactory implements FeatureFactory {
		
		protected final int dims;
		
		public ByteFeatureFactory(int dims) {
			this.dims = dims;			
			if(dims > 0xFFFF)
				throw new UnsupportedOperationException("Features with more than "+0xFFFF+" dimensions are not supported");
		}

		@Override
		public Class<?> getComponentType() {
			return byte.class;
		}

		@Override
		public int featureSize() {
			return dims;
		}

		@Override
		public int dims() {
			return dims;
		}

//		@Override
//		public FeatureVector of(byte[] featureBytes) {
//			final byte[] feature = new byte[dims];
//			ByteBuffer.wrap(featureBytes).get(feature);
//			return new ByteFeature(feature);
//		}

		@Override
		public FeatureVector read(DataInput is) throws IOException {
			final byte[] feature = new byte[dims];
			for (int i = 0; i < feature.length; i++) 
				feature[i] = is.readByte();
			return new ByteFeature(feature);
		}
	}
	
	/**
	 * Factory for the {@link ShortFeature}
	 * 
	 * @author Nico Hezel
	 */
	public static class ShortFeatureFactory implements FeatureFactory {
		
		protected final int dims;
		
		public ShortFeatureFactory(int dims) {
			this.dims = dims;			
			if(dims > 0xFFFF)
				throw new UnsupportedOperationException("Features with more than "+0xFFFF+" dimensions are not supported");
		}

		@Override
		public Class<?> getComponentType() {
			return short.class;
		}

		@Override
		public int featureSize() {
			return dims * Short.BYTES;
		}

		@Override
		public int dims() {
			return dims;
		}

//		@Override
//		public FeatureVector of(byte[] featureBytes) {
//			final short[] feature = new short[dims];
//			ByteBuffer.wrap(featureBytes).asShortBuffer().get(feature);
//			return new ShortFeature(feature);
//		}

		@Override
		public FeatureVector read(DataInput is) throws IOException {
			final short[] feature = new short[dims];
			for (int i = 0; i < feature.length; i++) 
				feature[i] = is.readShort();
			return new ShortFeature(feature);
		}
	}
	
	/**
	 * Factory for the {@link IntFeature}
	 * 
	 * @author Nico Hezel
	 */
	public static class IntFeatureFactory implements FeatureFactory {
		
		protected final int dims;
		
		public IntFeatureFactory(int dims) {
			this.dims = dims;			
			if(dims > 0xFFFF)
				throw new UnsupportedOperationException("Features with more than "+0xFFFF+" dimensions are not supported");
		}

		@Override
		public Class<?> getComponentType() {
			return int.class;
		}

		@Override
		public int featureSize() {
			return dims * Integer.BYTES;
		}

		@Override
		public int dims() {
			return dims;
		}

//		@Override
//		public FeatureVector of(byte[] featureBytes) {
//			final int[] feature = new int[dims];
//			ByteBuffer.wrap(featureBytes).asIntBuffer().get(feature);
//			return new IntFeature(feature);
//		}

		@Override
		public FeatureVector read(DataInput is) throws IOException {
			final int[] feature = new int[dims];
			for (int i = 0; i < feature.length; i++) 
				feature[i] = is.readInt();
			return new IntFeature(feature);
		}
	}
	
	/**
	 * Factory for the {@link LongFeature}
	 * 
	 * @author Nico Hezel
	 */
	public static class LongFeatureFactory implements FeatureFactory {
		
		protected final int dims;
		
		public LongFeatureFactory(int dims) {
			this.dims = dims;			
			if(dims > 0xFFFF)
				throw new UnsupportedOperationException("Features with more than "+0xFFFF+" dimensions are not supported");
		}

		@Override
		public Class<?> getComponentType() {
			return long.class;
		}

		@Override
		public int featureSize() {
			return dims * Long.BYTES;
		}

		@Override
		public int dims() {
			return dims;
		}

//		@Override
//		public FeatureVector of(byte[] featureBytes) {
//			final long[] feature = new long[dims];
//			ByteBuffer.wrap(featureBytes).asLongBuffer().get(feature);
//			return new LongFeature(feature);
//		}

		@Override
		public FeatureVector read(DataInput is) throws IOException {
			final long[] feature = new long[dims];
			for (int i = 0; i < feature.length; i++) 
				feature[i] = is.readLong();
			return new LongFeature(feature);
		}
	}

	/**
	 * Factory for the {@link FloatFeature}
	 * 
	 * @author Nico Hezel
	 */
	public static class FloatFeatureFactory implements FeatureFactory {
		
		protected final int dims;
		
		public FloatFeatureFactory(int dims) {
			this.dims = dims;			
			if(dims > 0xFFFF)
				throw new UnsupportedOperationException("Features with more than "+0xFFFF+" dimensions are not supported");
		}

		@Override
		public Class<?> getComponentType() {
			return float.class;
		}

		@Override
		public int featureSize() {
			return dims * Float.BYTES;
		}

		@Override
		public int dims() {
			return dims;
		}

//		@Override
//		public FeatureVector of(byte[] featureBytes) {
//			final float[] feature = new float[dims];
//			ByteBuffer.wrap(featureBytes).asFloatBuffer().get(feature);
//			return new FloatFeature(feature);
//		}

		@Override
		public FeatureVector read(DataInput is) throws IOException {
			final float[] feature = new float[dims];
			for (int i = 0; i < feature.length; i++) 
				feature[i] = is.readFloat();
			return new FloatFeature(feature);
		}
	}
	
	/**
	 * Factory for the {@link DoubleFeature}
	 * 
	 * @author Nico Hezel
	 */
	public static class DoubleFeatureFactory implements FeatureFactory {
		
		protected final int dims;
		
		public DoubleFeatureFactory(int dims) {
			this.dims = dims;			
			if(dims > 0xFFFF)
				throw new UnsupportedOperationException("Features with more than "+0xFFFF+" dimensions are not supported");
		}

		@Override
		public Class<?> getComponentType() {
			return double.class;
		}

		@Override
		public int featureSize() {
			return dims * Double.BYTES;
		}

		@Override
		public int dims() {
			return dims;
		}

//		@Override
//		public FeatureVector of(byte[] featureBytes) {
//			final double[] feature = new double[dims];
//			ByteBuffer.wrap(featureBytes).asDoubleBuffer().get(feature);
//			return new DoubleFeature(feature);
//		}

		@Override
		public FeatureVector read(DataInput is) throws IOException {
			final double[] feature = new double[dims];
			for (int i = 0; i < feature.length; i++) 
				feature[i] = is.readDouble();
			return new DoubleFeature(feature);
		}
	}	
}