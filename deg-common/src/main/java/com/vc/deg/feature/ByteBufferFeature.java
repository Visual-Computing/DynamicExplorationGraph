package com.vc.deg.feature;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import com.vc.deg.FeatureVector;

/**
 * Generic feature vector and contain all kinds of data 
 * 
 * @author Nico Hezel
 */
public class ByteBufferFeature implements FeatureVector {
	
	protected final ByteBuffer feature;
	
	/**
	 * Stores a copy of the byte-array
	 * 
	 * @param feature in ByteOrder.LITTLE_ENDIAN
	 */
	public ByteBufferFeature(byte[] feature) {
		this.feature = ByteBuffer.allocate(feature.length).order(ByteOrder.LITTLE_ENDIAN);
		this.feature.put(feature);
	}

	@Override
	public int size() {
		return feature.capacity();
	}

	@Override
	public boolean readBoolean(long index) {
		final int longIndex = (int)(index / (Long.BYTES * 8));
		final long mem = feature.getLong(longIndex);
		return ((mem << (index - longIndex)) & 0x1) != 0;
	}

	@Override
	public byte readByte(long index) {
		return feature.get((int)index * Byte.BYTES);
	}

	@Override
	public short readShort(long index) {
		return feature.getShort((int)index * Short.BYTES);
	}

	@Override
	public int readInt(long index) {
		return feature.getInt((int)index * Integer.BYTES);
	}

	@Override
	public long readLong(long index) {
		return feature.getLong((int)index * Long.BYTES);
	}

	@Override
	public float readFloat(long index) {
		return feature.getFloat((int)index * Float.BYTES);
	}

	@Override
	public double readDouble(long index) {
		return feature.getDouble((int)index * Double.BYTES);
	}

	@Override
	public byte[] toBytes() {
		final byte[] dst = new byte[size()];
		feature.rewind();
		feature.get(dst, 0, size());
		return dst;
	}
}