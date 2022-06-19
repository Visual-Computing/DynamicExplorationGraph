package com.vc.deg;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import com.vc.deg.feature.ByteFeature;
import com.vc.deg.feature.DoubleFeature;
import com.vc.deg.feature.FeatureFactory;
import com.vc.deg.feature.FloatFeature;
import com.vc.deg.feature.IntFeature;
import com.vc.deg.feature.ShortFeature;

/**
 * The {@link FeatureVector} is just a slice of memory containing feature values. 
 * The {@link FeatureSpace} knows how to read those values and compute the distance between two vectors.
 * Custom {@link FeatureVector} are classes which implement this interface but are not provided by the library.
 * For each custom {@link FeatureVector} there needs to be a custom {@link FeatureFactory}.
 * 
 * If the {@link FeatureVector} contains several features they can be accessed like they were placed 
 * in a consecutive in memory region. e.g. the first features uses 10 floats, than the second would start 
 * at a byte offset of 40. All methods use such an byte offset instead of an index parameter.
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
	 * Is the feature vector in native memory?
	 * 
	 * @return
	 */
	public boolean isNative();
	
	/**
	 * If this feature vector stores its value in native memory {@link #isNative()} == true, 
	 * than this address can be used to get access to the data.
	 * 
	 * @return
	 */
	public long nativeAddress();
	
	/**
	 * Read a byte from the index
	 * 
	 * @param byteOffset index at a specific byte position not array position
	 * @return
	 */
	public byte readByte(int byteOffset);
	
	/**
	 * Read a short from the index
	 * 
	 * @param byteOffset index at a specific byte position not array position
	 * @return
	 */
	public short readShort(int byteOffset);
	
	/**
	 * Read a int from the index
	 * 
	 * @param byteOffset index at a specific byte position not array position
	 * @return
	 */
	public int readInt(int byteOffset);
	
	/**
	 * Read a long from the index
	 * 
	 * @param byteOffset index at a specific byte position not array position
	 * @return
	 */
	public long readLong(int byteOffset);
	
	/**
	 * Read a float from the index
	 * 
	 * @param byteOffset index at a specific byte position not array position
	 * @return
	 */
	public float readFloat(int byteOffset);
	
	/**
	 * Read a double from the index
	 * 
	 * @param byteOffset index at a specific byte position not array position
	 * @return
	 */
	public double readDouble(int byteOffset);
	
	/**
	 * A copy of the feature vector in bytes (ByteOrder.LITTLE_ENDIAN)
	 * 
	 * @return
	 */
	public byte[] toBytes();
	
	/**
	 * Create a copy of the feature vector
	 * 
	 * @return
	 */
	public FeatureVector copy();
	
	/**
	 * Only the raw data should be written, no meta data like array size or type.
	 * Both are inferred by {@link FeatureFactory} which creates empty {@link FeatureVector} of the right type and size.
	 * 
	 * @param out
	 */
	public void writeObject(DataOutputStream out) throws IOException;
	
	/**
	 * Only the raw data should be read, no meta data like array size or type.
	 * Both are inferred by {@link FeatureFactory} which creates empty {@link FeatureVector} of the right type and size.
	 * 
	 * @param in
	 */
	public void readObject(DataInputStream in) throws IOException;
	
	
	
	// --------------------------------------------------------------------
	// ---------------- Primitive Array Wrapper methods -------------------
	// --------------------------------------------------------------------
	
	// TODO add array with offset and length support
	
	/**
	 * Wrap the array and create a feature vector with on-heap memory
	 * 
	 * @param feature
	 * @return
	 */
	public static FeatureVector wrap(byte[] feature) {
		return new ByteFeature(feature);
	}
	
	/**
	 * Wrap the array and create a feature vector with on-heap memory
	 * 
	 * @param feature
	 * @return
	 */
	public static FeatureVector wrap(short[] feature) {
		return new ShortFeature(feature);
	}
	
	/**
	 * Wrap the array and create a feature vector with on-heap memory
	 * 
	 * @param feature
	 * @return
	 */
	public static FeatureVector wrap(int[] feature) {
		return new IntFeature(feature);
	}
	
	/**
	 * Wrap the array and create a feature vector with on-heap memory
	 * 
	 * @param feature
	 * @return
	 */
	public static FeatureVector wrap(float[] feature) {
		return new FloatFeature(feature);
	}
	
	/**
	 * Wrap the array and create a feature vector with on-heap memory
	 * 
	 * @param feature
	 * @return
	 */
	public static FeatureVector wrap(double[] feature) {
		return new DoubleFeature(feature);
	}
	
	/**
	 * Register a custom feature factory to creating custom {@link FeatureVector}
	 *  
	 * @param featureFactory
	 */
	public static void registerFeatureFactor(FeatureFactory featureFactory) {
		throw new UnsupportedOperationException("registerin custom feature factories are not yet supported");
	}
}