package com.vc.deg.feature;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

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
		return feature[(int)index];		
	}

	@Override
	public byte[] toBytes() {
		final ByteBuffer bb = ByteBuffer.allocate(size()).order(ByteOrder.LITTLE_ENDIAN);
		for (double value : feature) 
			bb.putDouble(value);
		return bb.array();
	}
}