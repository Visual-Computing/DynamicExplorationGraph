package com.vc.deg.feature;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import com.vc.deg.FeatureVector;

/**
 * Wraps an float-array
 * 
 * @author Nico Hezel
 */
public class FloatFeature implements FeatureVector {
	
	protected final float[] feature;
	
	public FloatFeature(float[] feature) {
		this.feature = feature;
	}

	@Override
	public int size() {
		return feature.length * Float.BYTES;
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
		return feature[(int)index];
	}

	@Override
	public double readDouble(long index) {
		throw new UnsupportedOperationException("IntFeature does not support readDouble");
	}

	@Override
	public byte[] toBytes() {
		final ByteBuffer bb = ByteBuffer.allocate(size()).order(ByteOrder.LITTLE_ENDIAN);
		for (float value : feature) 
			bb.putFloat(value);
		return bb.array();
	}
}