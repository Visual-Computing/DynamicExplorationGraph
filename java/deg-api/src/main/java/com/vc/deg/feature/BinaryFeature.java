package com.vc.deg.feature;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;

import com.vc.deg.FeatureVector;

/**
 * The bits are represented in long, therefore is {@link #size()} in bytes always a multiple of 8.
 * The correct number of bits stored in this feature can be received by the {@link #dims()} method.
 * 
 * @author Nico Hezel
 *
 */
public class BinaryFeature implements FeatureVector {

	public static final Class<Binary> ComponentType = Binary.class;
	
	
	protected final long[] feature;
	protected final int dims;
	
	public BinaryFeature(int dims, long[] longFeatureVector) {
		this.feature = longFeatureVector;
		this.dims = dims;
	}
	
	/**
	 * Encode the byte array into longs which are used internal to represent the bits
	 * 
	 * @param byteFeatureVector
	 */
	public BinaryFeature(byte[] byteFeatureVector) {
		this.feature = encode(byteFeatureVector);
		this.dims = byteFeatureVector.length * 8;
	}

	/**
	 * Encode the byte array into longs 
	 * 
	 * @param input with signed bytes 
	 */
	protected long[] encode(byte[] input) {
		final int size = (int) Math.ceil((float)input.length / 8);
		final ByteBuffer bb = ByteBuffer.wrap(Arrays.copyOf(input, size * 8));
		
		final long[] result = new long[size];
		bb.asLongBuffer().get(result, 0, result.length);
		return result;
	}
	
	public long get(int index) {
		return feature[index];
	}
	
	@Override
	public Class<?> getComponentType() {
		return ComponentType;
	}

	@Override
	public int dims() {
		return dims;
	}
	
	@Override
	public int size() {
		return feature.length * Long.BYTES;
	}

	@Override
	public boolean isNative() {
		return false;
	}

	@Override
	public byte readByte(int byteOffset) {
		throw new UnsupportedOperationException(BinaryFeature.class.getSimpleName() + " does not support readByte");
	}

	@Override
	public short readShort(int byteOffset) {
		throw new UnsupportedOperationException(BinaryFeature.class.getSimpleName() + " does not support readShort");
	}

	@Override
	public int readInt(int byteOffset) {
		throw new UnsupportedOperationException(BinaryFeature.class.getSimpleName() + " does not support readInt");
	}

	@Override
	public long readLong(int byteOffset) {
		return feature[byteOffset >> 3];
	}

	@Override
	public float readFloat(int byteOffset) {
		throw new UnsupportedOperationException(BinaryFeature.class.getSimpleName() + " does not support readFloat");
	}

	@Override
	public double readDouble(int byteOffset) {
		throw new UnsupportedOperationException(BinaryFeature.class.getSimpleName() + " does not support readDouble");
	}
	
	@Override
	public FeatureVector copy() {
		return new BinaryFeature(dims, Arrays.copyOf(feature, feature.length));
	}
	
	@Override
	public void writeObject(DataOutput out) throws IOException {
		for (long d : feature) 
			out.writeLong(d);
	}

	@Override
	public void readObject(DataInput in) throws IOException {		
		for (int i = 0; i < feature.length; i++) 
			feature[i] = in.readLong();
	}

	@Override
	public long nativeAddress() {
		throw new UnsupportedOperationException(BinaryFeature.class.getSimpleName() + " stores its values on-heap, using a native address is dangerous.");
	}
	


	// dummy class to represent an internal component of this feature vector
	public static class Binary {}
}