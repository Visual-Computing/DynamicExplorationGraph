package com.vc.deg.feature;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import com.vc.deg.FeatureVector;

/**
 * Generic feature vector and contains all kinds of data 
 * 
 * TODO A Buffer Version is not needed, but a Chronical Bytes with offheap memory 
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
	public boolean isNative() {
		return false;
	}

	@Override
	public byte readByte(int byteOffset) {
		return feature.get(byteOffset);
	}

	@Override
	public short readShort(int byteOffset) {
		return feature.getShort(byteOffset);
	}

	@Override
	public int readInt(int byteOffset) {
		return feature.getInt(byteOffset);
	}

	@Override
	public long readLong(int byteOffset) {
		return feature.getLong(byteOffset);
	}

	@Override
	public float readFloat(int byteOffset) {
		return feature.getFloat(byteOffset);
	}

	@Override
	public double readDouble(int byteOffset) {
		return feature.getDouble(byteOffset);
	}

	@Override
	public byte[] toBytes() {
		final byte[] dst = new byte[size()];
		feature.rewind();
		feature.get(dst, 0, size());
		return dst;
	}
	
	@Override
	public FeatureVector copy() {
		return new ByteBufferFeature(toBytes());
	}

	@Override
	public long nativeAddress() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void writeObject(DataOutputStream out) throws IOException {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void readObject(DataInputStream in) throws IOException {
		// TODO Auto-generated method stub
		
	}
}