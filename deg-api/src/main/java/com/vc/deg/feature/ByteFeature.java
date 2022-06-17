package com.vc.deg.feature;

import java.util.Arrays;

import com.vc.deg.FeatureVector;

/**
 * Wraps an double-array
 * 
 * @author Nico Hezel
 */
public class ByteFeature implements FeatureVector {
	
	protected final byte[] feature;
	
	public ByteFeature(byte[] feature) {
		this.feature = feature;
	}

	@Override
	public int size() {
		return feature.length;
	}

	@Override
	public boolean readBoolean(long index) {
		throw new UnsupportedOperationException("IntFeature does not support readBoolean");
	}

	@Override
	public byte readByte(long index) {
		return feature[(int)index];
	}

	@Override
	public short readShort(long index) {
		throw new UnsupportedOperationException("IntFeature does not support readShort");
	}

	@Override
	public int readInt(long index) {
		throw new UnsupportedOperationException("IntFeature does not support readInt");
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
		return Arrays.copyOf(feature, feature.length);
	}
}