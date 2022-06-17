package com.vc.deg;

import com.vc.deg.feature.BooleanFeature;
import com.vc.deg.feature.ByteFeature;
import com.vc.deg.feature.DoubleFeature;
import com.vc.deg.feature.FloatFeature;
import com.vc.deg.feature.IntFeature;
import com.vc.deg.feature.ShortFeature;

/**
 * The feature vector is just a slice of memory. 
 * The feature space know how to read it and compute the distance between two vectors.
 * 
 * @author Nico Hezel
 */
public interface FeatureVector {
	
	/**
	 * Size in bytes
	 * 
	 * @return
	 */
	public int size();
	
	/**
	 * Read a boolean (single bit) from the index 
	 * 
	 * @param index
	 * @return
	 */
	public boolean readBoolean(long index);
	
	/**
	 * Read a byte from the index
	 * 
	 * @param index
	 * @return
	 */
	public byte readByte(long index);
	
	/**
	 * Read a short from the index
	 * 
	 * @param index
	 * @return
	 */
	public short readShort(long index);
	
	/**
	 * Read a int from the index
	 * 
	 * @param index
	 * @return
	 */
	public int readInt(long index);
	
	/**
	 * Read a long from the index
	 * 
	 * @param index
	 * @return
	 */
	public long readLong(long index);
	
	/**
	 * Read a float from the index
	 * 
	 * @param index
	 * @return
	 */
	public float readFloat(long index);
	
	/**
	 * Read a double from the index
	 * 
	 * @param index
	 * @return
	 */
	public double readDouble(long index);
	
	/**
	 * A copy of the feature vector in bytes (ByteOrder.LITTLE_ENDIAN)
	 * 
	 * @return
	 */
	public byte[] toBytes();	
	
	
	
	// --------------------------------------------------------------------
	// ---------------- Primitive Array Wrapper methods -------------------
	// --------------------------------------------------------------------
	
	/**
	 * 
	 * @param feature
	 * @return
	 */
	public static FeatureVector wrap(byte[] feature) {
		return new ByteFeature(feature);
	}
	
	/**
	 * 
	 * @param feature
	 * @return
	 */
	public static FeatureVector wrap(short[] feature) {
		return new ShortFeature(feature);
	}
	
	/**
	 * 
	 * @param feature
	 * @return
	 */
	public static FeatureVector wrap(int[] feature) {
		return new IntFeature(feature);
	}
	
	/**
	 * 
	 * @param feature
	 * @return
	 */
	public static FeatureVector wrap(float[] feature) {
		return new FloatFeature(feature);
	}
	
	/**
	 * 
	 * @param feature
	 * @return
	 */
	public static FeatureVector wrap(double[] feature) {
		return new DoubleFeature(feature);
	}
	
	/**
	 * 
	 * @param feature
	 * @return
	 */
	public static FeatureVector wrapBits(byte[] bitFeature) {
		return new BooleanFeature(bitFeature);
	}
	
	/**
	 * 
	 * @param feature
	 * @return
	 */
	public static FeatureVector wrapBits(long[] bitFeature) {
		return new BooleanFeature(bitFeature);
	}
}