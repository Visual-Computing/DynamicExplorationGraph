package com.vc.deg.feature;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

import com.vc.deg.FeatureVector;

/**
 * Wraps an short-array
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
	public byte readByte(int byteOffset) {
		throw new UnsupportedOperationException(ShortFeature.class.getSimpleName() + " does not support readByte");
	}

	@Override
	public short readShort(int byteOffset) {
		return feature[byteOffset >> 1];
	}

	@Override
	public int readInt(int byteOffset) {
		throw new UnsupportedOperationException(ShortFeature.class.getSimpleName() + " does not support readInt");
	}

	@Override
	public long readLong(int byteOffset) {
		throw new UnsupportedOperationException(ShortFeature.class.getSimpleName() + " does not support readLong");
	}

	@Override
	public float readFloat(int byteOffset) {
		throw new UnsupportedOperationException(ShortFeature.class.getSimpleName() + " does not support readFloat");
	}

	@Override
	public double readDouble(int byteOffset) {
		throw new UnsupportedOperationException(ShortFeature.class.getSimpleName() + " does not support readDouble");
	}

	@Override
	public byte[] toBytes() {
		final ByteBuffer bb = ByteBuffer.allocate(size()).order(ByteOrder.LITTLE_ENDIAN);
		for (short value : feature) 
			bb.putShort(value);
		return bb.array();
	}
	
	@Override
	public FeatureVector copy() {
		return new ShortFeature(Arrays.copyOf(feature, feature.length));
	}
	
	@Override
	public void writeObject(DataOutputStream out) throws IOException {
		for (short d : feature) 
			out.writeShort(d);
	}

	@Override
	public void readObject(DataInputStream in) throws IOException {
		for (int i = 0; i < feature.length; i++) 
			feature[i] = in.readShort();
	}

	@Override
	public long nativeAddress() {
		throw new UnsupportedOperationException(ShortFeature.class.getSimpleName() + " stores its values on-heap, using a native address is dangerous.");
	}
	
	@Override
	public boolean isNative() {
		return false;
	}
}