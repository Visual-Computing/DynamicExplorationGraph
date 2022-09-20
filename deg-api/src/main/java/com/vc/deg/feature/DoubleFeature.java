package com.vc.deg.feature;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

import com.vc.deg.FeatureVector;

/**
 * Wraps an double-array
 * 
 * @author Nico Hezel
 */
public class DoubleFeature implements FeatureVector {
	
	protected final double[] feature;
	
	public DoubleFeature(double[] feature) {
		this.feature = feature;
	}

	@Override
	public int size() {
		return feature.length * Double.BYTES;
	}

	@Override
	public byte readByte(int byteOffset) {
		throw new UnsupportedOperationException(DoubleFeature.class.getSimpleName() + " does not support readByte");
	}

	@Override
	public short readShort(int byteOffset) {
		throw new UnsupportedOperationException(DoubleFeature.class.getSimpleName() + " does not support readShort");
	}

	@Override
	public int readInt(int byteOffset) {
		throw new UnsupportedOperationException(DoubleFeature.class.getSimpleName() + " does not support readInt");
	}

	@Override
	public long readLong(int byteOffset) {
		throw new UnsupportedOperationException(DoubleFeature.class.getSimpleName() + " does not support readLong");
	}

	@Override
	public float readFloat(int byteOffset) {
		throw new UnsupportedOperationException(DoubleFeature.class.getSimpleName() + " does not support readFloat");
	}

	@Override
	public double readDouble(int byteOffset) {
		return feature[byteOffset >> 3];		
	}

	@Override
	public byte[] toBytes() {
		final ByteBuffer bb = ByteBuffer.allocate(size()).order(ByteOrder.LITTLE_ENDIAN);
		for (double value : feature) 
			bb.putDouble(value);
		return bb.array();
	}
	
	@Override
	public FeatureVector copy() {
		return new DoubleFeature(Arrays.copyOf(feature, feature.length));
	}
	
	@Override
	public void writeObject(DataOutput out) throws IOException {
		for (double d : feature) 
			out.writeDouble(d);
	}

	@Override
	public void readObject(DataInput in) throws IOException {
		for (int i = 0; i < feature.length; i++) 
			feature[i] = in.readDouble();
	}

	@Override
	public long nativeAddress() {
		throw new UnsupportedOperationException(DoubleFeature.class.getSimpleName() + " stores its values on-heap, using a native address is dangerous.");
	}
	
	@Override
	public boolean isNative() {
		return false;
	}
}