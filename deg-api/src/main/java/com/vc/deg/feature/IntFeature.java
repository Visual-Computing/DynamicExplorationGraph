package com.vc.deg.feature;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import com.vc.deg.FeatureVector;

/**
 * Wraps an int-array
 * 
 * @author Nico Hezel
 */
public class IntFeature implements FeatureVector {
	
	protected final int[] feature;
	
	public IntFeature(int[] feature) {
		this.feature = feature;
	}

	@Override
	public int size() {
		return feature.length * Integer.BYTES;
	}

	@Override
	public boolean readBoolean(long index) {
		throw new UnsupportedOperationException("IntFeature does not support readBoolean");
	}

	@Override
	public byte readByte(long index) {
		throw new UnsupportedOperationException("IntFeature does not support readByte");
	}

	@Override
	public short readShort(long index) {
		throw new UnsupportedOperationException("IntFeature does not support readShort");
	}

	@Override
	public int readInt(long index) {
		return feature[(int)index];
	}

	@Override
	public long readLong(long index) {
		throw new UnsupportedOperationException("IntFeature does not support readLong");
	}

	@Override
	public float readFloat(long index) {
		throw new UnsupportedOperationException("IntFeature does not support readFloat");
	}

	@Override
	public double readDouble(long index) {
		throw new UnsupportedOperationException("IntFeature does not support readDouble");
	}

	@Override
	public byte[] toBytes() {
		final ByteBuffer bb = ByteBuffer.allocate(size()).order(ByteOrder.LITTLE_ENDIAN);
		for (int value : feature) 
			bb.putInt(value);
		return bb.array();
	}
}