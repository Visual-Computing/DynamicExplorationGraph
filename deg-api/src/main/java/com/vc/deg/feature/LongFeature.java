package com.vc.deg.feature;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

import com.vc.deg.FeatureVector;

/**
 * Wraps an long-array
 * 
 * @author Nico Hezel
 */
public class LongFeature implements FeatureVector {
	
	protected final long[] feature;
	
	public LongFeature(long[] feature) {
		this.feature = feature;
	}
	
	public long get(int index) {
		return feature[index];
	}

	@Override
	public int size() {
		return feature.length * Long.BYTES;
	}

	@Override
	public byte readByte(int byteOffset) {
		throw new UnsupportedOperationException(LongFeature.class.getSimpleName() + " does not support readByte");
	}

	@Override
	public short readShort(int byteOffset) {
		throw new UnsupportedOperationException(LongFeature.class.getSimpleName() + " does not support readShort");
	}

	@Override
	public int readInt(int byteOffset) {
		throw new UnsupportedOperationException(LongFeature.class.getSimpleName() + " does not support readInt");
	}

	@Override
	public long readLong(int byteOffset) {
		return feature[byteOffset >> 3];
	}

	@Override
	public float readFloat(int byteOffset) {
		throw new UnsupportedOperationException(LongFeature.class.getSimpleName() + " does not support readFloat");
	}

	@Override
	public double readDouble(int byteOffset) {
		throw new UnsupportedOperationException(LongFeature.class.getSimpleName() + " does not support readDouble");
	}

	@Override
	public byte[] toBytes() {
		final ByteBuffer bb = ByteBuffer.allocate(size()).order(ByteOrder.LITTLE_ENDIAN);
		for (long value : feature) 
			bb.putLong(value);
		return bb.array();
	}
	
	@Override
	public FeatureVector copy() {
		return new LongFeature(Arrays.copyOf(feature, feature.length));
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
		throw new UnsupportedOperationException(LongFeature.class.getSimpleName() + " stores its values on-heap, using a native address is dangerous.");
	}
	
	@Override
	public boolean isNative() {
		return false;
	}
}