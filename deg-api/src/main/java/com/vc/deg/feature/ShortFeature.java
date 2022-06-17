package com.vc.deg.feature;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import com.vc.deg.FeatureVector;

/**
 * Wraps an int-array
 * 
 * @author Nico Hezel
 */
public class ShortFeature implements FeatureVector {
	
	protected final short[] feature;
	
	public ShortFeature(short[] feature) {
		this.feature = feature;
	}

	@Override
	public int size() {
		return feature.length * Short.BYTES;
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
		return feature[(int)index];
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
		final ByteBuffer bb = ByteBuffer.allocate(size()).order(ByteOrder.LITTLE_ENDIAN);
		for (short value : feature) 
			bb.putShort(value);
		return bb.array();
	}
}